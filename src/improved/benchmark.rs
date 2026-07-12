// NEW ! TO UNDERSTAND : this file centralizes ALL benchmark-related functions
// (Sanity Checks, the full sumcheck protocol benchmark, and offline/online timing),
// so that main.rs only orchestrates top-level calls.

use ark_test_curves::bls12_381::Fr;
use ark_ff::{UniformRand, Field, PrimeField};
use ark_linear_sumcheck::ml_sumcheck::MLSumcheck;
use ark_poly::MultilinearExtension;
use std::fs::{File, OpenOptions};

use std::time::{Duration, Instant};
use std::sync::atomic::Ordering;
use std::io::{Write, stdout};

use crate::improved::arithmetic::{FAST_PATH_COUNT, SLOW_PATH_COUNT, adaptive_dot_product_accumulate, extrapolate_dot_product}; // Adjust path according to your project structure
use crate::improved::protocol::{EvalProductSV, LinearTimeSC, SumcheckProtocol};
use crate::improved::streaming::MockStream;
use crate::utils::{generate_multivariate_poly_test, generate_small_value_poly_test};
// NEW ! TO UNDERSTAND : the tracking global allocator is declared once in main.rs
// (`#[global_allocator] pub static PEAK_ALLOC: PeakAlloc = ...`); we just read it here.
use crate::PEAK_ALLOC;

// NEW ! TO UNDERSTAND : shared sweep parameters, so that run_all_sc_benchmark and
// bench_offline_seq_vs_parallel cover the exact same (Variables, Degree) grid.
const MAX_VARS: usize = 14;
const NUM_RUNS: u32 = 3;
const DEGREES_TO_TEST: [usize; 5] = [2, 3, 4, 6, 8]; // Extended range of degrees for a smooth 3D surface

// NEW ! TO UNDERSTAND : memory is far more deterministic than wall-clock time (no OS/CPU
// scheduling noise), so a single run per (Variables, Degree) point is enough — this keeps
// the already-heavy full sweep (up to num_vars=14) from taking even longer.
const NUM_RUNS_MEMORY: u32 = 1;

// =================================================================================================
// 1. SANITY CHECK 1 : MULTIPLICATION RATIO BENCHMARK
// =================================================================================================

/// Sanity Check 1: Measures the performance profile 
/// comparing the precomputed extrapolate_dot_product
/// on both full-size random fields (Big) and controlled integers (Small).
pub fn run_multiplication_ratio_benchmark() {
    let mut rng = ark_std::test_rng();
    let size = 1_000_000;

    println!("Running Comprehensive Sanity Check 1 Matrix...");

    // 1. Setup inputs
    let big_elements: Vec<Fr> = (0..size).map(|_| Fr::rand(&mut rng)).collect();
    let small_elements: Vec<Fr> = (0..size).map(|i| Fr::from((i % 5 + 1) as u64)).collect();
    let coefficients: Vec<Fr> = (0..size).map(|_| Fr::rand(&mut rng)).collect();
    
    // Precompute limbs required by the new extrapolate function interface
    let coeff_limbs: Vec<_> = coefficients.iter().map(|c| c.into_bigint()).collect();

    // =========================================================================
    // COMBINATION 3: New Extrapolate + Big Elements (Checks fallback mechanism overhead)
    // =========================================================================
    let mut acc_extrapolate_big = Fr::ZERO;
    let start_3 = Instant::now();
    extrapolate_dot_product(&mut acc_extrapolate_big, &big_elements, &coeff_limbs, &coefficients);
    let dur_extrapolate_big = start_3.elapsed().as_secs_f64() * 1000.0;

    // =========================================================================
    // COMBINATION 4: New Extrapolate + Small Elements (Pure optimized Fast-Path)
    // =========================================================================
    let mut acc_extrapolate_small = Fr::ZERO;
    let start_4 = Instant::now();
    extrapolate_dot_product(&mut acc_extrapolate_small, &small_elements, &coeff_limbs, &coefficients);
    let dur_extrapolate_small = start_4.elapsed().as_secs_f64() * 1000.0;

    // Print the benchmark summary directly to the terminal
    println!("------------------------------------------------------------");
    println!("| Configuration                               | Time (ms)  |");
    println!("------------------------------------------------------------");
    println!("| Extrapolate Precomputed (Big Elements)      | {:10.4} |", dur_extrapolate_big);
    println!("| Extrapolate Precomputed (Small Elements)    | {:10.4} |", dur_extrapolate_small);
    println!("------------------------------------------------------------");

    // Save to the CSV file. Note: The python plotting tool will automatically capture 
    // these 4 discrete categories.
    let mut file = File::create("csv/multiplication_ratio.csv").expect("Unable to create ratio file");
    writeln!(file, "Operation,Time_ms").unwrap();
    writeln!(file, "Extrapolate (Big Elements),{:.4}", dur_extrapolate_big).unwrap();
    writeln!(file, "Extrapolate (Small Elements),{:.4}", dur_extrapolate_small).unwrap();
    file.flush().unwrap();
}

// =================================================================================================
// 2. FULL SUMCHECK PROTOCOL BENCHMARK (Arkworks vs LinearTimeSC vs EvalProductSV)
// =================================================================================================

/// NEW ! TO UNDERSTAND : this function used to be the body of `main()` in main.rs.
/// It now owns the whole "3D" sumcheck protocol benchmark: it sweeps over degrees and
/// number of variables, and writes everything to csv/benchmark_3d_data.csv.
pub fn run_all_sc_benchmark() {
    let max_vars = MAX_VARS;
    let num_runs = NUM_RUNS;
    let degrees_to_test = DEGREES_TO_TEST;

    println!("==================================================");
    println!("       STARTING SUMCHECK PROTOCOL BENCHMARK        ");
    println!("==================================================");

    // Initialize the global 3D benchmark file
    let global_filename = "csv/benchmark_3d_data.csv";
    let mut file = File::create(global_filename).expect("Unable to create global file");
    writeln!(
        file,
        "Variables,Degree,Arkworks_ms,LinearTime_Vanilla_ms,LinearTime_SB1_ms,EvalProductSV_Total_ms,EvalProductSV_Offline_ms,EvalProductSV_Online_ms"
    ).unwrap();
    drop(file); // Close to avoid borrow issues, append mode will be used later

    for &d in &degrees_to_test {
        println!("\n##################################################");
        println!("  LAUNCHING BENCHMARK SERIES FOR DEGREE d = {}", d);
        println!("##################################################");
        test_range_variables_3d(max_vars, d, num_runs, global_filename);
    }

    println!("\n[GLOBAL OK] All benchmarks completed successfully!");
}

pub fn test_range_variables_3d(max_vars: usize, d: usize, num_runs: u32, out_filename: &str) {
    let mut file = OpenOptions::new()
        .write(true)
        .append(true)
        .open(out_filename)
        .expect("Unable to open data file in append mode");

    for num_v in 4..=max_vars {
        println!("\n==================================================");
        println!(
            " Benchmarking for {} variables (2^{} = {} points, d={})",
            num_v, num_v, 1 << num_v, d
        );
        println!(" Average over {} runs...", num_runs);
        println!("==================================================");

        let mut total_ark = Duration::ZERO;
        let mut total_vanilla = Duration::ZERO;
        let mut total_sb1 = Duration::ZERO;
        let mut total_sv_offline = Duration::ZERO;
        let mut total_sv_online = Duration::ZERO;

        for run in 1..=num_runs {
            print!("   Run {}/{}... ", run, num_runs);
            stdout().flush().unwrap();

            let (d_ark, d_vanilla, d_sb1, d_sv_offline, d_sv_online) = multivariate_test(num_v, d);

            total_ark += d_ark;
            total_vanilla += d_vanilla;
            total_sb1 += d_sb1;
            total_sv_offline += d_sv_offline;
            total_sv_online += d_sv_online;

            println!("Done.");
        }

        let avg_ark = total_ark / num_runs;
        let avg_vanilla = total_vanilla / num_runs;
        let avg_sb1 = total_sb1 / num_runs;
        let avg_sv_offline = total_sv_offline / num_runs;
        let avg_sv_online = total_sv_online / num_runs;
        let avg_sv_total = avg_sv_offline + avg_sv_online;

        let duration_ark_ms = avg_ark.as_secs_f64() * 1000.0;
        let duration_vanilla_ms = avg_vanilla.as_secs_f64() * 1000.0;
        let duration_sb1_ms = avg_sb1.as_secs_f64() * 1000.0;
        let duration_sv_offline_ms = avg_sv_offline.as_secs_f64() * 1000.0;
        let duration_sv_online_ms = avg_sv_online.as_secs_f64() * 1000.0;
        let duration_sv_total_ms = avg_sv_total.as_secs_f64() * 1000.0;

        // Save structured entry for 3D engine plotting [X=Variables, Y=Degree, Z=Times...]
        writeln!(
            file,
            "{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}",
            num_v, d, duration_ark_ms, duration_vanilla_ms, duration_sb1_ms, duration_sv_total_ms, duration_sv_offline_ms, duration_sv_online_ms
        )
        .unwrap();
        file.flush().unwrap();
    }
}

fn multivariate_test(num_vars: usize, d: usize) -> (Duration, Duration, Duration, Duration, Duration) {
    let mut rng = rand::thread_rng();
    
    // --- SETUP SELECTOR FOR SANITY CHECK 0 ---
    // Standard setup with full-size random field elements (Fast-path rate will be 0.00%):
    // let (list_of_poly, list_of_products) = generate_multivariate_poly_test(&mut rng, num_vars, d);
    
    // Optimized small-value setting setup (Triggers the custom fast-path branches):
    let (list_of_poly, list_of_products) = generate_small_value_poly_test(&mut rng, num_vars, d);
    //let (list_of_poly, list_of_products) = generate_multivariate_poly_test(&mut rng, num_vars, d);
    // -----------------------------------------

    let hypercube_size = 1 << num_vars;
    let mut expected_sum = Fr::ZERO;
    for x in 0..hypercube_size {
        let mut product_at_x = Fr::ONE;
        for k in 0..d {
            product_at_x *= list_of_poly[k].evaluations[x];
        }
        expected_sum += product_at_x;
    }

    let start_arkworks = Instant::now();
    let proof = MLSumcheck::prove(&list_of_products).expect("The Arkworks prover failed");
    let duration_arkworks = start_arkworks.elapsed();

    let ark_sum = MLSumcheck::extract_sum(&proof);
    assert_eq!(expected_sum, ark_sum, "Local hypercube sum mismatch!");

    let linear_time_protocol_vanilla = LinearTimeSC;
    let l = list_of_poly[0].num_vars();
    let d_len = list_of_poly.len();
    let mut stream_opt = MockStream::new(l, d_len, &list_of_poly);

    let start_vanilla = Instant::now();
    let verifier_accepted_vanilla = linear_time_protocol_vanilla.run(&mut stream_opt, expected_sum);
    let duration_vanilla = start_vanilla.elapsed();
    assert!(verifier_accepted_vanilla);

    let linear_time_protocol_sb1 = LinearTimeSC;
    let start_sb1 = Instant::now();
    let verifier_accepted_sb1 = linear_time_protocol_sb1.run_sb_1(&mut stream_opt, expected_sum);
    let duration_sb1 = start_sb1.elapsed();
    assert!(verifier_accepted_sb1);

    let eval_product_sv_protocol = EvalProductSV::new(d_len, l);
    let mut stream_sv = MockStream::new(l, d_len, &list_of_poly);

    // Measure the Offline Phase (Pure geometric precomputation)
    let start_offline = Instant::now();
    let offline_data = eval_product_sv_protocol.precomputation_phase(&mut stream_sv);
    let duration_sv_offline = start_offline.elapsed();

    // Measure the Online Phase (Rounds simulation + Final Phase with interaction)
    let start_online = Instant::now();
    let verifier_accepted_sv = eval_product_sv_protocol.online_phase(&mut stream_sv, expected_sum, offline_data);
    let duration_sv_online = start_online.elapsed();
    assert!(verifier_accepted_sv);

    // --- SANITY CHECK 0 INTEGRATION ---
    // Print stats and automatically reset counters for the next variable iteration
    println!("\n[STATS] Evaluation results for num_vars = {} and degree d = {}:", num_vars, d);
    print_and_reset_arithmetic_counters(); 

    (duration_arkworks, duration_vanilla, duration_sb1, duration_sv_offline, duration_sv_online)
}

// =================================================================================================
// 3. SANITY CHECK 0 : ADAPTIVE ARITHMETIC COUNTERS
// =================================================================================================

/// Prints the current state of the adaptive arithmetic counters and resets them to zero.
/// Used for Sanity Check 0 to verify if the fast-path (Small-Big) is triggered.
pub fn print_and_reset_arithmetic_counters() {
    let fast = FAST_PATH_COUNT.swap(0, Ordering::Relaxed);
    let slow = SLOW_PATH_COUNT.swap(0, Ordering::Relaxed);
    let total = fast + slow;

    println!("--------------------------------------------------");
    println!("       SANITY CHECK 0: ARITHMETIC COUNTERS        ");
    println!("--------------------------------------------------");
    println!(" -> Fast-Path Calls (Small-Big Mul): {}", fast);
    println!(" -> Slow-Path Calls (Big-Big Mul)  : {}", slow);
    if total > 0 {
        let ratio = (fast as f64 / total as f64) * 100.0;
        println!(" -> Fast-Path Utilization Rate     : {:.2}%", ratio);
    } else {
        println!(" -> Fast-Path Utilization Rate     : 0.00% (No operations executed)");
    }
    println!("--------------------------------------------------");
}

// =================================================================================================
// 4. OFFLINE PHASE : SEQUENTIAL VS PARALLEL BENCHMARK
// =================================================================================================

/// NEW ! TO UNDERSTAND : bench_offline_seq_vs_parallel now sweeps over the same
/// (Degree, Variables) grid as run_all_sc_benchmark (same DEGREES_TO_TEST / MAX_VARS /
/// NUM_RUNS constants), and measures BOTH the parallel and the sequential offline
/// precomputation phase for each (d, l) pair. Results are written to
/// csv/offline_seq_vs_parallel.csv so they can be plotted the same way.
pub fn bench_offline_seq_vs_parallel() {
    println!("==================================================");
    println!("   STARTING OFFLINE PRECOMPUTATION BENCHMARK       ");
    println!("   (Sequential vs Parallel - EvalProductSV)        ");
    println!("==================================================");

    let filename = "csv/offline_seq_vs_parallel.csv";
    let mut file = File::create(filename).expect("Unable to create offline benchmark file");
    writeln!(file, "Variables,Degree,Offline_Sequential_ms,Offline_Parallel_ms").unwrap();
    drop(file); // Close to avoid borrow issues, append mode will be used below

    for &d in &DEGREES_TO_TEST {
        println!("\n##################################################");
        println!("  OFFLINE BENCHMARK SERIES FOR DEGREE d = {}", d);
        println!("##################################################");

        let mut file = OpenOptions::new()
            .write(true)
            .append(true)
            .open(filename)
            .expect("Unable to open offline benchmark file in append mode");

        let l = 14;
        let mut rng = rand::thread_rng();
        let (list_of_poly, _) = generate_small_value_poly_test(&mut rng, l, d);
        let protocol = EvalProductSV::new(d, l);

        let mut total_sequential = Duration::ZERO;
        let mut total_parallel = Duration::ZERO;

        for run in 1..=NUM_RUNS {
            print!("   d={d} l={l} run {run}/{NUM_RUNS}... ");
            stdout().flush().unwrap();

            // Sequential offline precomputation
            let mut stream_seq = MockStream::new(l, d, &list_of_poly);
            let start_seq = Instant::now();
            let _ = protocol.precomputation_phase_sequential(&mut stream_seq);
            total_sequential += start_seq.elapsed();

            // Parallel offline precomputation (current default implementation)
            let mut stream_par = MockStream::new(l, d, &list_of_poly);
            let start_par = Instant::now();
            let _ = protocol.precomputation_phase(&mut stream_par);
            total_parallel += start_par.elapsed();

            println!("Done.");
        }

        let avg_sequential_ms = (total_sequential / NUM_RUNS).as_secs_f64() * 1000.0;
        let avg_parallel_ms = (total_parallel / NUM_RUNS).as_secs_f64() * 1000.0;
        println!(
            "d={d} l={l} : average sequential offline = {:.4} ms | average parallel offline = {:.4} ms",
            avg_sequential_ms, avg_parallel_ms
        );

        writeln!(file, "{},{},{:.4},{:.4}", l, d, avg_sequential_ms, avg_parallel_ms).unwrap();
        file.flush().unwrap();
    }

    println!("\n[OFFLINE OK] All offline precomputation benchmarks completed successfully!");
}

// =================================================================================================
// NEW ! TO UNDERSTAND
// 5. MEMORY BENCHMARK (Arkworks vs LinearTimeSC vs EvalProductSV)
//    Same (Variables, Degree) grid as run_all_sc_benchmark, but measuring PEAK HEAP MEMORY
//    (bytes allocated on top of whatever was already resident) instead of wall-clock time.
//    Relies on `PEAK_ALLOC` (a `peak_alloc::PeakAlloc` global allocator installed in main.rs)
//    to observe allocations/deallocations happening anywhere in the measured closure,
//    including inside rayon worker threads.
// =================================================================================================

/// Runs `f`, returning both its result and the PEAK extra number of bytes that were
/// allocated (and not yet freed) at any point during the call, relative to whatever was
/// already allocated right before the call started.
///
/// Note: because `PEAK_ALLOC` is a single process-wide allocator, this must be called with
/// no other benchmark measurement running concurrently on another thread — which holds here
/// since the whole benchmark suite runs sequentially from `main()`.
fn measure_peak_bytes<T>(f: impl FnOnce() -> T) -> (T, usize) {
    let baseline = PEAK_ALLOC.current_usage();
    PEAK_ALLOC.reset_peak_usage();
    let result = f();
    let peak = PEAK_ALLOC.peak_usage();
    (result, peak.saturating_sub(baseline))
}

/// NEW ! TO UNDERSTAND : memory equivalent of `run_all_sc_benchmark`. Sweeps the same
/// (Variables, Degree) grid and writes csv/benchmark_3d_memory_data.csv.
pub fn run_all_sc_memory_benchmark() {
    let max_vars = MAX_VARS;
    let num_runs = NUM_RUNS_MEMORY;
    let degrees_to_test = DEGREES_TO_TEST;

    println!("==================================================");
    println!("        STARTING SUMCHECK MEMORY BENCHMARK         ");
    println!("==================================================");

    let global_filename = "csv/benchmark_3d_memory_data.csv";
    let mut file = File::create(global_filename).expect("Unable to create global memory file");
    writeln!(
        file,
        "Variables,Degree,Arkworks_KB,LinearTime_Vanilla_KB,LinearTime_SB1_KB,EvalProductSV_Total_KB,EvalProductSV_Offline_KB,EvalProductSV_Online_KB"
    ).unwrap();
    drop(file);

    for &d in &degrees_to_test {
        println!("\n##################################################");
        println!("  LAUNCHING MEMORY BENCHMARK SERIES FOR DEGREE d = {}", d);
        println!("##################################################");
        test_range_variables_3d_memory(max_vars, d, num_runs, global_filename);
    }

    println!("\n[GLOBAL OK] All memory benchmarks completed successfully!");
}

pub fn test_range_variables_3d_memory(max_vars: usize, d: usize, num_runs: u32, out_filename: &str) {
    let mut file = OpenOptions::new()
        .write(true)
        .append(true)
        .open(out_filename)
        .expect("Unable to open memory data file in append mode");

    for num_v in 4..=max_vars {
        println!("\n==================================================");
        println!(
            " Measuring memory for {} variables (2^{} = {} points, d={})",
            num_v, num_v, 1 << num_v, d
        );
        println!(" Average over {} run(s)...", num_runs);
        println!("==================================================");

        let mut total_ark: usize = 0;
        let mut total_vanilla: usize = 0;
        let mut total_sb1: usize = 0;
        let mut total_sv_total: usize = 0;
        let mut total_sv_offline: usize = 0;
        let mut total_sv_online: usize = 0;

        for run in 1..=num_runs {
            print!("   Run {}/{}... ", run, num_runs);
            stdout().flush().unwrap();

            let (m_ark, m_vanilla, m_sb1, m_sv_total, m_sv_offline, m_sv_online) =
                multivariate_memory_test(num_v, d);

            total_ark += m_ark;
            total_vanilla += m_vanilla;
            total_sb1 += m_sb1;
            total_sv_total += m_sv_total;
            total_sv_offline += m_sv_offline;
            total_sv_online += m_sv_online;

            println!("Done.");
        }

        let num_runs_usize = num_runs as usize;
        let avg_ark_kb = (total_ark / num_runs_usize) as f64 / 1024.0;
        let avg_vanilla_kb = (total_vanilla / num_runs_usize) as f64 / 1024.0;
        let avg_sb1_kb = (total_sb1 / num_runs_usize) as f64 / 1024.0;
        let avg_sv_total_kb = (total_sv_total / num_runs_usize) as f64 / 1024.0;
        let avg_sv_offline_kb = (total_sv_offline / num_runs_usize) as f64 / 1024.0;
        let avg_sv_online_kb = (total_sv_online / num_runs_usize) as f64 / 1024.0;

        writeln!(
            file,
            "{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}",
            num_v, d, avg_ark_kb, avg_vanilla_kb, avg_sb1_kb, avg_sv_total_kb, avg_sv_offline_kb, avg_sv_online_kb
        )
        .unwrap();
        file.flush().unwrap();
    }
}

/// NEW ! TO UNDERSTAND : memory equivalent of `multivariate_test`. Runs each protocol once
/// and reports the peak extra heap bytes allocated during that specific call, using
/// `measure_peak_bytes`. Mirrors the structure of `multivariate_test` closely on purpose so
/// both benchmarks stay easy to compare / keep in sync.
fn multivariate_memory_test(num_vars: usize, d: usize) -> (usize, usize, usize, usize, usize, usize) {
    let mut rng = rand::thread_rng();

    let (list_of_poly, list_of_products) = generate_small_value_poly_test(&mut rng, num_vars, d);

    let hypercube_size = 1 << num_vars;
    let mut expected_sum = Fr::ZERO;
    for x in 0..hypercube_size {
        let mut product_at_x = Fr::ONE;
        for k in 0..d {
            product_at_x *= list_of_poly[k].evaluations[x];
        }
        expected_sum += product_at_x;
    }

    // --- Arkworks ---
    let (proof, mem_ark) = measure_peak_bytes(|| {
        MLSumcheck::prove(&list_of_products).expect("The Arkworks prover failed")
    });
    let ark_sum = MLSumcheck::extract_sum(&proof);
    assert_eq!(expected_sum, ark_sum, "Local hypercube sum mismatch!");

    let l = list_of_poly[0].num_vars();
    let d_len = list_of_poly.len();

    // --- LinearTimeSC Vanilla ---
    let linear_time_protocol_vanilla = LinearTimeSC;
    let mut stream_vanilla = MockStream::new(l, d_len, &list_of_poly);
    let (accepted_vanilla, mem_vanilla) = measure_peak_bytes(|| {
        linear_time_protocol_vanilla.run(&mut stream_vanilla, expected_sum)
    });
    assert!(accepted_vanilla);

    // --- LinearTimeSC SB1 ---
    let linear_time_protocol_sb1 = LinearTimeSC;
    let mut stream_sb1 = MockStream::new(l, d_len, &list_of_poly);
    let (accepted_sb1, mem_sb1) = measure_peak_bytes(|| {
        linear_time_protocol_sb1.run_sb_1(&mut stream_sb1, expected_sum)
    });
    assert!(accepted_sb1);

    let eval_product_sv_protocol = EvalProductSV::new(d_len, l);

    // --- EvalProductSV: Total (Offline + Online run back-to-back in ONE measured window,
    //     so this reflects the real combined peak footprint rather than the sum of two
    //     separately-measured peaks, which could be misleading if they don't overlap). ---
    let mut stream_sv_total = MockStream::new(l, d_len, &list_of_poly);
    let (accepted_sv_total, mem_sv_total) = measure_peak_bytes(|| {
        eval_product_sv_protocol.run(&mut stream_sv_total, expected_sum)
    });
    assert!(accepted_sv_total);

    // --- EvalProductSV: Offline phase ONLY ---
    let mut stream_sv_offline = MockStream::new(l, d_len, &list_of_poly);
    let (_offline_data_unused, mem_sv_offline) = measure_peak_bytes(|| {
        eval_product_sv_protocol.precomputation_phase(&mut stream_sv_offline)
    });

    // --- EvalProductSV: Online phase ONLY (its own offline precomputation is run OUTSIDE
    //     the measured window, so the reported peak reflects only what the online phase
    //     itself needs on top of an already-available offline grid). ---
    let mut stream_sv_online = MockStream::new(l, d_len, &list_of_poly);
    let offline_data_for_online = eval_product_sv_protocol.precomputation_phase(&mut stream_sv_online);
    let (accepted_sv_online, mem_sv_online) = measure_peak_bytes(|| {
        eval_product_sv_protocol.online_phase(&mut stream_sv_online, expected_sum, offline_data_for_online)
    });
    assert!(accepted_sv_online);

    println!(
        "[MEM] num_vars={} d={} | Arkworks={:.2} KB | Vanilla={:.2} KB | SB1={:.2} KB | SV_Total={:.2} KB | SV_Offline={:.2} KB | SV_Online={:.2} KB",
        num_vars, d,
        mem_ark as f64 / 1024.0,
        mem_vanilla as f64 / 1024.0,
        mem_sb1 as f64 / 1024.0,
        mem_sv_total as f64 / 1024.0,
        mem_sv_offline as f64 / 1024.0,
        mem_sv_online as f64 / 1024.0,
    );

    (mem_ark, mem_vanilla, mem_sb1, mem_sv_total, mem_sv_offline, mem_sv_online)
}
