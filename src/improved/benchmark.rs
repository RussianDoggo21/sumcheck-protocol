// NEW ! TO UNDERSTAND : this file centralizes ALL benchmark-related functions
// (Sanity Checks, the full sumcheck protocol benchmark, and offline/online timing),
// so that main.rs only orchestrates top-level calls.

use ark_test_curves::bls12_381::Fr;
use ark_ff::{UniformRand, Field, PrimeField, BigInteger384, BigInteger256};
use ark_linear_sumcheck::ml_sumcheck::MLSumcheck;
use ark_poly::MultilinearExtension;
use std::fs::{File, OpenOptions};

use std::time::{Duration, Instant};
use std::sync::atomic::Ordering;
use std::io::{Write, stdout};

use crate::improved::arithmetic::{
    FAST_PATH_COUNT, SLOW_PATH_COUNT,
    adaptive_dot_product_accumulate, extrapolate_dot_product,
    small_big_raw, finalize_delayed_reduction,
};
use crate::improved::protocol::{EvalProductSV, LinearTimeSC, SumcheckProtocol};
use crate::improved::streaming::MockStream;
use crate::utils::{generate_multivariate_poly_test, generate_small_value_poly_test};
// NEW ! TO UNDERSTAND : the tracking global allocator is declared once in main.rs
// (`#[global_allocator] pub static PEAK_ALLOC: PeakAlloc = ...`); we just read it here.
use crate::PEAK_ALLOC;

// NEW ! TO UNDERSTAND : shared sweep parameters, so that run_all_sc_benchmark and
// bench_run_seq_vs_parallel cover the exact same (Variables, Degree) grid.
const MAX_VARS: usize = 14;
const NUM_RUNS: u32 = 3;
const DEGREES_TO_TEST: [usize; 5] = [2, 3, 4, 6, 8]; // Extended range of degrees for a smooth 3D surface

// NEW ! TO UNDERSTAND : memory is far more deterministic than wall-clock time (no OS/CPU
// scheduling noise), so a single run per (Variables, Degree) point is enough — this keeps
// the already-heavy full sweep (up to num_vars=14) from taking even longer.
const NUM_RUNS_MEMORY: u32 = 1;

// =================================================================================================
// 1. SANITY CHECK 1 : MULTIPLICATION RATIO BENCHMARKS (batch + solo)
// =================================================================================================

/// NEW ! TO UNDERSTAND : orchestrator. Runs both the batch (dot-product) comparison and the
/// solo (single multiplication) comparison, one after the other. main.rs keeps calling this
/// single function; nothing changes on its side.
pub fn run_multiplication_ratio_benchmark() {
    run_batch_multiplication_benchmark();
    run_solo_multiplication_benchmark();
}

/// Sanity Check 1 (batch): measures the cost of a full dot product (1,000,000 terms in a
/// single call) under several code paths, on two operand regimes (big/random vs small-value).
fn run_batch_multiplication_benchmark() {
    let mut rng = ark_std::test_rng();
    let size = 1_000_000;

    println!("Running Sanity Check 1 (batch): dot product over {size} terms...");

    let big_elements: Vec<Fr> = (0..size).map(|_| Fr::rand(&mut rng)).collect();
    let small_elements: Vec<Fr> = (0..size).map(|_| Fr::from(10u64)).collect();
    let coefficients: Vec<Fr> = (0..size).map(|_| Fr::rand(&mut rng)).collect();
    let coeff_limbs: Vec<_> = coefficients.iter().map(|c| c.into_bigint()).collect();

    // Naive per-term multiply-accumulate loop (plain Fr * Fr), used as the "obvious,
    // unoptimized" baseline -- this is the code you would write if you had never heard of
    // small-value arithmetic.
    let mut acc_naive_big = Fr::ZERO;
    let start = Instant::now();
    for i in 0..size {
        acc_naive_big += big_elements[i] * coefficients[i];
    }
    let dur_naive_big = start.elapsed().as_secs_f64() * 1000.0;

    let mut acc_naive_small = Fr::ZERO;
    let start = Instant::now();
    for i in 0..size {
        acc_naive_small += small_elements[i] * coefficients[i];
    }
    let dur_naive_small = start.elapsed().as_secs_f64() * 1000.0;

    // NEW ! TO UNDERSTAND : mirrors the solo benchmark's structure at batch scale --
    // window_evals' bigints are ALSO precomputed outside the timed loop (like coeff_limbs
    // already are), and the loop assumes every element is small (true here, by construction
    // of small_elements) so it can skip the per-term into_bigint()/smallness check entirely
    // and just run small_big_raw, finalizing once at the very end. This isolates the
    // delayed-reduction technique's true achievable throughput, stripped of the per-term
    // detection cost that extrapolate_dot_product necessarily pays in the general case (it
    // cannot know in advance which elements are small).
    let start0 =  Instant::now();
    let small_bigints: Vec<BigInteger256> = small_elements.iter().map(|e| e.into_bigint()).collect();
    let mut acc_precomputed_small = Fr::ZERO;
    let start = Instant::now();
    let mut global_t = BigInteger384::zero();
    for i in 0..size {
        let small = small_bigints[i].0[0]; // guaranteed small by construction of small_elements
        small_big_raw(&mut global_t, small, &coeff_limbs[i]);
    }
    acc_precomputed_small += finalize_delayed_reduction(&global_t);
    let dur_precomputed_small = start.elapsed().as_secs_f64() * 1000.0;
    let dur_small = start0.elapsed().as_secs_f64() * 1000.0;

    assert_eq!(acc_naive_small, acc_precomputed_small, "Mathematical mismatch on Precomputed Small!");

    println!("------------------------------------------------------------");
    println!("| Configuration                                     | Time (ms) |");
    println!("------------------------------------------------------------");
    println!("| Naive (Big Elements)                              | {:10.4}   |", dur_naive_big);
    println!("| Naive (Small Elements)                            | {:10.4}   |", dur_naive_small);
    println!("| Small-big multiplication (bigints precomputed)    | {:10.4}   |", dur_precomputed_small);
    println!("| Small-big multiplication (no precomputation)    | {:10.4}   |", dur_small);
    println!("------------------------------------------------------------");

    let mut file = File::create("csv/multiplication_ratio_batch.csv").expect("Unable to create batch ratio file");
    writeln!(file, "Operation,Time_ms").unwrap();
    writeln!(file, "Naive (Big Elements),{:.4}", dur_naive_big).unwrap();
    writeln!(file, "Naive (Small Elements),{:.4}", dur_naive_small).unwrap();
    writeln!(file, "Small-big (Small bigints precomputed),{:.4}", dur_precomputed_small).unwrap();
    writeln!(file, "Small-big (No precomputation),{:.4}", dur_small).unwrap();
    file.flush().unwrap();
}

/// Sanity Check 1 bis (solo). Measures the cost of a SINGLE multiplication `small * big`
/// under four different code paths: sb / bb / arkworks / small_big_raw (isolated).
fn run_solo_multiplication_benchmark() {
    let mut rng = ark_std::test_rng();
    let size = 1_000_000;

    println!("Running Sanity Check 1 bis (solo): single multiplication, 4 code paths...");

    let small: u64 = 13;
    let small_as_fr = Fr::from(small);
    let bigs: Vec<Fr> = (0..size).map(|_| Fr::rand(&mut rng)).collect();
    let bigs2: Vec<Fr> = (0..size).map(|_| Fr::rand(&mut rng)).collect();

    for i in 0..1000 {
        let expected = small_as_fr * bigs[i];
        let mut global_t = BigInteger384::zero();
        small_big_raw(&mut global_t, small, &bigs[i].into_bigint());
        assert_eq!(finalize_delayed_reduction(&global_t), expected, "small_big_raw mismatch");
    }

    let mut sink = Fr::ZERO;

    let start = Instant::now();
    for i in 0..size { sink += bigs[i] * bigs2[i]; }
    sink = std::hint::black_box(sink);
    let dur_bb = start.elapsed().as_secs_f64() * 1e9 / size as f64;

    let start = Instant::now();
    for i in 0..size { sink += small_as_fr * bigs[i]; }
    sink = std::hint::black_box(sink);
    let dur_arkworks = start.elapsed().as_secs_f64() * 1e9 / size as f64;

    let start0 = Instant::now();
    let bigints: Vec<BigInteger256> = bigs.iter().map(|b| b.into_bigint()).collect();

    let start = Instant::now();
    for i in 0..size {
        let mut global_t = BigInteger384::zero();
        small_big_raw(&mut global_t, small, &bigints[i]);
        std::hint::black_box(&global_t);
    }
    let dur_mac = start.elapsed().as_secs_f64() * 1e9 / size as f64;
    let dur_mac0 = start0.elapsed().as_secs_f64() * 1e9 / size as f64;

    println!("------------------------------------------------------------");
    println!("| Configuration                    | Time (ns/call)      |");
    println!("------------------------------------------------------------");
    println!("| bb (native big * big)            | {:10.2}             |", dur_bb);
    println!("| arkworks (Fr::from(small) * big) | {:10.2}             |", dur_arkworks);
    println!("| small_big_raw (without conversion and reduction)    | {:10.2}             |", dur_mac);
    println!("| small_big_raw (with conversion and reduction)    | {:10.2}             |", dur_mac0);
    println!("------------------------------------------------------------");
    assert!(sink != Fr::ZERO, "sink was optimized away -- benchmark results are meaningless");

    let mut file = File::create("csv/multiplication_ratio_solo.csv").expect("Unable to create solo ratio file");
    writeln!(file, "Operation,Time_ns").unwrap();
    writeln!(file, "bb (native big * big),{:.4}", dur_bb).unwrap();
    writeln!(file, "arkworks (Fr::from(small) * big),{:.4}", dur_arkworks).unwrap();
    writeln!(file, "small_big_raw (without conversion and reduction),{:.4}", dur_mac).unwrap();
    writeln!(file, "small_big_raw (with conversion and reduction),{:.4}", dur_mac0).unwrap();
    file.flush().unwrap();
}

// =================================================================================================
// 2. FULL SUMCHECK PROTOCOL BENCHMARK (Arkworks vs LinearTimeSC vs EvalProductSV)
// =================================================================================================

/// This function used to be the body of `main()` in main.rs. It now owns the whole "3D"
/// sumcheck protocol benchmark: it sweeps over degrees and number of variables, and writes
/// everything to csv/benchmark_3d_data.csv.
pub fn run_all_sc_benchmark() {
    let max_vars = MAX_VARS;
    let num_runs = NUM_RUNS;
    let degrees_to_test = DEGREES_TO_TEST;

    println!("==================================================");
    println!("       STARTING SUMCHECK PROTOCOL BENCHMARK        ");
    println!("==================================================");

    // NEW ! TO UNDERSTAND : EvalProductSV_Offline_ms/EvalProductSV_Online_ms columns removed --
    // EvalProductSV is now called via a single `run()`, so there is only one timing left for it.
    let global_filename = "csv/benchmark_3d_data.csv";
    let mut file = File::create(global_filename).expect("Unable to create global file");
    writeln!(
        file,
        "Variables,Degree,Arkworks_ms,LinearTime_Vanilla_ms,LinearTime_SB1_ms,EvalProductSV_Total_ms"
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
        let mut total_sv = Duration::ZERO;

        for run in 1..=num_runs {
            print!("   Run {}/{}... ", run, num_runs);
            stdout().flush().unwrap();

            let (d_ark, d_vanilla, d_sb1, d_sv) = multivariate_test(num_v, d);

            total_ark += d_ark;
            total_vanilla += d_vanilla;
            total_sb1 += d_sb1;
            total_sv += d_sv;

            println!("Done.");
        }

        let avg_ark = total_ark / num_runs;
        let avg_vanilla = total_vanilla / num_runs;
        let avg_sb1 = total_sb1 / num_runs;
        let avg_sv = total_sv / num_runs;

        let duration_ark_ms = avg_ark.as_secs_f64() * 1000.0;
        let duration_vanilla_ms = avg_vanilla.as_secs_f64() * 1000.0;
        let duration_sb1_ms = avg_sb1.as_secs_f64() * 1000.0;
        let duration_sv_ms = avg_sv.as_secs_f64() * 1000.0;

        // Save structured entry for 3D engine plotting [X=Variables, Y=Degree, Z=Times...]
        writeln!(
            file,
            "{},{},{:.4},{:.4},{:.4},{:.4}",
            num_v, d, duration_ark_ms, duration_vanilla_ms, duration_sb1_ms, duration_sv_ms
        )
        .unwrap();
        file.flush().unwrap();
    }
}

fn multivariate_test(num_vars: usize, d: usize) -> (Duration, Duration, Duration, Duration) {
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

    // NEW ! TO UNDERSTAND : EvalProductSV is now a single call -- no more separate
    // precomputation_phase/online_phase timings, per your tutor's note.
    let eval_product_sv_protocol = EvalProductSV::new(d_len, l);
    let mut stream_sv = MockStream::new(l, d_len, &list_of_poly);

    let start_sv = Instant::now();
    let verifier_accepted_sv = eval_product_sv_protocol.run(&mut stream_sv, expected_sum);
    let duration_sv = start_sv.elapsed();
    assert!(verifier_accepted_sv);

    // --- SANITY CHECK 0 INTEGRATION ---
    // Print stats and automatically reset counters for the next variable iteration
    println!("\n[STATS] Evaluation results for num_vars = {} and degree d = {}:", num_vars, d);
    print_and_reset_arithmetic_counters();

    (duration_arkworks, duration_vanilla, duration_sb1, duration_sv)
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
// 4. WHOLE-PROTOCOL SEQUENTIAL VS PARALLEL BENCHMARK
// =================================================================================================

/// NEW ! TO UNDERSTAND : renamed from bench_offline_seq_vs_parallel. Since EvalProductSV no
/// longer exposes a separate "offline phase", this now compares the two whole-protocol entry
/// points instead: `run_sequential` (grid built with a plain for-loop) vs `run` (grid built
/// with rayon). The "online" part (interactive rounds + final linear descent) is identical,
/// unparallelized code in both, so it contributes equally to both timings and doesn't distort
/// the comparison -- what moves is exactly the benefit of parallelizing the grid construction.
/// Sweeps the same (Degree, Variables) grid as run_all_sc_benchmark. Results are written to
/// csv/run_seq_vs_parallel.csv.
pub fn bench_run_seq_vs_parallel() {
    println!("==================================================");
    println!("   STARTING WHOLE-PROTOCOL BENCHMARK               ");
    println!("   (Sequential vs Parallel - EvalProductSV::run)   ");
    println!("==================================================");

    let filename = "csv/run_seq_vs_parallel.csv";
    let mut file = File::create(filename).expect("Unable to create run seq/parallel benchmark file");
    writeln!(file, "Variables,Degree,Run_Sequential_ms,Run_Parallel_ms").unwrap();
    drop(file); // Close to avoid borrow issues, append mode will be used below

    for &d in &DEGREES_TO_TEST {
        println!("\n##################################################");
        println!("  RUN BENCHMARK SERIES FOR DEGREE d = {}", d);
        println!("##################################################");

        let mut file = OpenOptions::new()
            .write(true)
            .append(true)
            .open(filename)
            .expect("Unable to open run seq/parallel benchmark file in append mode");

        let l = 14;
        let mut rng = rand::thread_rng();
        let (list_of_poly, _) = generate_small_value_poly_test(&mut rng, l, d);
        let protocol = EvalProductSV::new(d, l);

        let mut total_sequential = Duration::ZERO;
        let mut total_parallel = Duration::ZERO;

        for run in 1..=NUM_RUNS {
            print!("   d={d} l={l} run {run}/{NUM_RUNS}... ");
            stdout().flush().unwrap();

            let hypercube_size = 1 << l;
            let mut expected_sum = Fr::ZERO;
            for x in 0..hypercube_size {
                let mut product_at_x = Fr::ONE;
                for k in 0..d { product_at_x *= list_of_poly[k].evaluations[x]; }
                expected_sum += product_at_x;
            }

            // Sequential whole-protocol run
            let mut stream_seq = MockStream::new(l, d, &list_of_poly);
            let start_seq = Instant::now();
            let accepted_seq = protocol.run_sequential(&mut stream_seq, expected_sum);
            total_sequential += start_seq.elapsed();
            assert!(accepted_seq);

            // Parallel whole-protocol run (current default implementation)
            let mut stream_par = MockStream::new(l, d, &list_of_poly);
            let start_par = Instant::now();
            let accepted_par = protocol.run(&mut stream_par, expected_sum);
            total_parallel += start_par.elapsed();
            assert!(accepted_par);

            println!("Done.");
        }

        let avg_sequential_ms = (total_sequential / NUM_RUNS).as_secs_f64() * 1000.0;
        let avg_parallel_ms = (total_parallel / NUM_RUNS).as_secs_f64() * 1000.0;
        println!(
            "d={d} l={l} : average sequential run = {:.4} ms | average parallel run = {:.4} ms",
            avg_sequential_ms, avg_parallel_ms
        );

        writeln!(file, "{},{},{:.4},{:.4}", l, d, avg_sequential_ms, avg_parallel_ms).unwrap();
        file.flush().unwrap();
    }

    println!("\n[RUN OK] All sequential vs parallel benchmarks completed successfully!");
}

// =================================================================================================
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

/// Memory equivalent of `run_all_sc_benchmark`. Sweeps the same (Variables, Degree) grid and
/// writes csv/benchmark_3d_memory_data.csv.
pub fn run_all_sc_memory_benchmark() {
    let max_vars = MAX_VARS;
    let num_runs = NUM_RUNS_MEMORY;
    let degrees_to_test = DEGREES_TO_TEST;

    println!("==================================================");
    println!("        STARTING SUMCHECK MEMORY BENCHMARK         ");
    println!("==================================================");

    // NEW ! TO UNDERSTAND : EvalProductSV_Offline_KB/EvalProductSV_Online_KB columns removed,
    // same reason as in run_all_sc_benchmark -- EvalProductSV is a single call now.
    let global_filename = "csv/benchmark_3d_memory_data.csv";
    let mut file = File::create(global_filename).expect("Unable to create global memory file");
    writeln!(
        file,
        "Variables,Degree,Arkworks_KB,LinearTime_Vanilla_KB,LinearTime_SB1_KB,EvalProductSV_Total_KB"
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
        let mut total_sv: usize = 0;

        for run in 1..=num_runs {
            print!("   Run {}/{}... ", run, num_runs);
            stdout().flush().unwrap();

            let (m_ark, m_vanilla, m_sb1, m_sv) = multivariate_memory_test(num_v, d);

            total_ark += m_ark;
            total_vanilla += m_vanilla;
            total_sb1 += m_sb1;
            total_sv += m_sv;

            println!("Done.");
        }

        let num_runs_usize = num_runs as usize;
        let avg_ark_kb = (total_ark / num_runs_usize) as f64 / 1024.0;
        let avg_vanilla_kb = (total_vanilla / num_runs_usize) as f64 / 1024.0;
        let avg_sb1_kb = (total_sb1 / num_runs_usize) as f64 / 1024.0;
        let avg_sv_kb = (total_sv / num_runs_usize) as f64 / 1024.0;

        writeln!(
            file,
            "{},{},{:.4},{:.4},{:.4},{:.4}",
            num_v, d, avg_ark_kb, avg_vanilla_kb, avg_sb1_kb, avg_sv_kb
        )
        .unwrap();
        file.flush().unwrap();
    }
}

/// Memory equivalent of `multivariate_test`. Runs each protocol once and reports the peak
/// extra heap bytes allocated during that specific call, using `measure_peak_bytes`.
///
/// NEW ! TO UNDERSTAND : EvalProductSV is now measured as a single `run()` call -- the
/// separate offline-only / online-only memory measurements were removed along with
/// precomputation_phase/online_phase.
fn multivariate_memory_test(num_vars: usize, d: usize) -> (usize, usize, usize, usize) {
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

    // --- EvalProductSV: single run() call ---
    let eval_product_sv_protocol = EvalProductSV::new(d_len, l);
    let mut stream_sv = MockStream::new(l, d_len, &list_of_poly);
    let (accepted_sv, mem_sv) = measure_peak_bytes(|| {
        eval_product_sv_protocol.run(&mut stream_sv, expected_sum)
    });
    assert!(accepted_sv);

    println!(
        "[MEM] num_vars={} d={} | Arkworks={:.2} KB | Vanilla={:.2} KB | SB1={:.2} KB | EvalProductSV={:.2} KB",
        num_vars, d,
        mem_ark as f64 / 1024.0,
        mem_vanilla as f64 / 1024.0,
        mem_sb1 as f64 / 1024.0,
        mem_sv as f64 / 1024.0,
    );

    (mem_ark, mem_vanilla, mem_sb1, mem_sv)
}
// =================================================================================================
// NEW ! TO UNDERSTAND
// 6. BIGINT FIELD SANITY CHECK: full-protocol vanilla vs 1-sb vs sb-all, using StdFr2
//    (bigint_field.rs / bigint_sumcheck.rs) instead of arkworks' Montgomery-only Fr.
//    This is the integration requested by Christopher (NAIST tutor): the same delayed
//    small-big technique as Sanity Check 1, but tested end-to-end, on a field
//    representation that never leaves standard form, to see whether the technique's
//    theoretical speedup (confirmed on the isolated primitive, see Sanity Check 1 bis)
//    survives once wired into the real round-by-round protocol.
// =================================================================================================

use crate::improved::bigint_sumcheck::{
    bigint_linear_time_sc_sb1, bigint_linear_time_sc_sb_all, bigint_linear_time_sc_vanilla,
};

pub fn bench_bigint_vanilla_vs_sb() {
    println!("==================================================");
    println!("  BIGINT FIELD SANITY CHECK (vanilla vs 1-sb vs sb-all)  ");
    println!("==================================================");

    // NEW ! TO UNDERSTAND : now sweeps the SAME (Degree, Variables) grid as
    // run_all_sc_benchmark (DEGREES_TO_TEST x 4..=MAX_VARS) instead of a fixed
    // [3, 6, 9] at Variables=14 only -- this is what lets plot_benchmarks.py overlay
    // BigInt curves directly onto the existing per-degree comparative plots
    // (sumcheck_benchmark_curve_d{d}.png), which are indexed by this exact grid.
    let filename = "csv/bigint_vanilla_vs_sb.csv";
    let mut file = File::create(filename).expect("Unable to create bigint sanity check file");
    writeln!(file, "Variables,Degree,Vanilla_ms,SB1_ms,SBAll_ms").unwrap();

    for &d in &DEGREES_TO_TEST {
        for num_vars in 4..=MAX_VARS {
            let mut rng = rand::thread_rng();

            let mut total_vanilla = Duration::ZERO;
            let mut total_sb1 = Duration::ZERO;
            let mut total_sb_all = Duration::ZERO;

            for run in 1..=NUM_RUNS {
                print!("   d={d} num_vars={num_vars} run {run}/{NUM_RUNS}... ");
                stdout().flush().unwrap();

                let (list_of_poly, _) = generate_multivariate_poly_test(&mut rng, num_vars, d);

                let hypercube_size = 1 << num_vars;
                let mut expected_sum = Fr::ZERO;
                for x in 0..hypercube_size {
                    let mut product_at_x = Fr::ONE;
                    for k in 0..d {
                        product_at_x *= list_of_poly[k].evaluations[x];
                    }
                    expected_sum += product_at_x;
                }

                let mut stream_vanilla = MockStream::new(num_vars, d, &list_of_poly);
                let start = Instant::now();
                let ok_vanilla = bigint_linear_time_sc_vanilla(&mut stream_vanilla, expected_sum);
                total_vanilla += start.elapsed();

                let mut stream_sb1 = MockStream::new(num_vars, d, &list_of_poly);
                let start = Instant::now();
                let ok_sb1 = bigint_linear_time_sc_sb1(&mut stream_sb1, expected_sum);
                total_sb1 += start.elapsed();

                let mut stream_sb_all = MockStream::new(num_vars, d, &list_of_poly);
                let start = Instant::now();
                let ok_sb_all = bigint_linear_time_sc_sb_all(&mut stream_sb_all, expected_sum);
                total_sb_all += start.elapsed();

                assert!(ok_vanilla, "bigint vanilla REJECTED for d={d} num_vars={num_vars}");
                assert!(ok_sb1, "bigint 1-sb REJECTED for d={d} num_vars={num_vars}");
                assert!(ok_sb_all, "bigint sb-all REJECTED for d={d} num_vars={num_vars}");

                println!("Done.");
            }

            let avg_vanilla_ms = (total_vanilla / NUM_RUNS).as_secs_f64() * 1000.0;
            let avg_sb1_ms = (total_sb1 / NUM_RUNS).as_secs_f64() * 1000.0;
            let avg_sb_all_ms = (total_sb_all / NUM_RUNS).as_secs_f64() * 1000.0;

            println!(
                "d={d:<2} num_vars={num_vars:<2} : vanilla = {avg_vanilla_ms:8.3} ms | 1-sb = {avg_sb1_ms:8.3} ms ({:.3}x) | sb-all = {avg_sb_all_ms:8.3} ms ({:.3}x)",
                avg_vanilla_ms / avg_sb1_ms, avg_vanilla_ms / avg_sb_all_ms
            );

            writeln!(file, "{},{},{:.4},{:.4},{:.4}", num_vars, d, avg_vanilla_ms, avg_sb1_ms, avg_sb_all_ms).unwrap();
            file.flush().unwrap();
        }
    }

    println!("\n[BIGINT OK] All configurations accepted by the verifier -- results written to csv/bigint_vanilla_vs_sb.csv");
}

/// NEW ! TO UNDERSTAND : memory equivalent of bench_bigint_vanilla_vs_sb, mirroring
/// run_all_sc_memory_benchmark's structure (single run per point, peak_alloc-based) --
/// lets plot_benchmarks.py overlay BigInt curves onto sumcheck_memory_curve_d{d}.png.
pub fn bench_bigint_memory() {
    println!("==================================================");
    println!("  BIGINT FIELD MEMORY BENCHMARK (vanilla vs 1-sb vs sb-all)  ");
    println!("==================================================");

    let filename = "csv/bigint_memory.csv";
    let mut file = File::create(filename).expect("Unable to create bigint memory file");
    writeln!(file, "Variables,Degree,Vanilla_KB,SB1_KB,SBAll_KB").unwrap();

    for &d in &DEGREES_TO_TEST {
        for num_vars in 4..=MAX_VARS {
            print!("   [MEM][BIGINT] d={d} num_vars={num_vars}... ");
            stdout().flush().unwrap();

            let mut rng = rand::thread_rng();
            let (list_of_poly, _) = generate_multivariate_poly_test(&mut rng, num_vars, d);

            let hypercube_size = 1 << num_vars;
            let mut expected_sum = Fr::ZERO;
            for x in 0..hypercube_size {
                let mut product_at_x = Fr::ONE;
                for k in 0..d {
                    product_at_x *= list_of_poly[k].evaluations[x];
                }
                expected_sum += product_at_x;
            }

            let mut stream_vanilla = MockStream::new(num_vars, d, &list_of_poly);
            let (ok_vanilla, mem_vanilla) = measure_peak_bytes(|| {
                bigint_linear_time_sc_vanilla(&mut stream_vanilla, expected_sum)
            });
            assert!(ok_vanilla, "bigint vanilla REJECTED for d={d} num_vars={num_vars}");

            let mut stream_sb1 = MockStream::new(num_vars, d, &list_of_poly);
            let (ok_sb1, mem_sb1) = measure_peak_bytes(|| {
                bigint_linear_time_sc_sb1(&mut stream_sb1, expected_sum)
            });
            assert!(ok_sb1, "bigint 1-sb REJECTED for d={d} num_vars={num_vars}");

            let mut stream_sb_all = MockStream::new(num_vars, d, &list_of_poly);
            let (ok_sb_all, mem_sb_all) = measure_peak_bytes(|| {
                bigint_linear_time_sc_sb_all(&mut stream_sb_all, expected_sum)
            });
            assert!(ok_sb_all, "bigint sb-all REJECTED for d={d} num_vars={num_vars}");

            let vanilla_kb = mem_vanilla as f64 / 1024.0;
            let sb1_kb = mem_sb1 as f64 / 1024.0;
            let sb_all_kb = mem_sb_all as f64 / 1024.0;

            println!("Vanilla={vanilla_kb:.2} KB | 1-sb={sb1_kb:.2} KB | sb-all={sb_all_kb:.2} KB");

            writeln!(file, "{},{},{:.4},{:.4},{:.4}", num_vars, d, vanilla_kb, sb1_kb, sb_all_kb).unwrap();
            file.flush().unwrap();
        }
    }

    println!("\n[BIGINT MEM OK] results written to csv/bigint_memory.csv");
}

// =================================================================================================
// NEW ! TO UNDERSTAND
// 7. STDFR2::MUL_BB VS ARKWORKS FR*FR (plus mul_sb for reference)
//    Isolates exactly how much of the BigInt-vs-arkworks gap seen in the curve overlays
//    of Sections 1/5 is attributable to our hand-written Barrett reduction (mul_bb) being
//    less mature than arkworks' assembly-optimized Montgomery multiplication -- as opposed
//    to anything about the small-value technique itself, which is what Sanity Check "BigInt
//    vanilla vs 1-sb vs sb-all" already isolates correctly (same field throughout).
// =================================================================================================

use crate::improved::bigint_field::StdFr2;

pub fn bench_mul_bb_vs_arkworks() {
    println!("==================================================");
    println!("  STDFR2::MUL_BB VS ARKWORKS FR*FR (+ MUL_SB REFERENCE)  ");
    println!("==================================================");

    let mut rng = rand::thread_rng();
    let size = 1_000_000;

    let bigs_a: Vec<Fr> = (0..size).map(|_| Fr::rand(&mut rng)).collect();
    let bigs_b: Vec<Fr> = (0..size).map(|_| Fr::rand(&mut rng)).collect();
    let smalls: Vec<u64> = (0..size).map(|_| rand::Rng::gen_range(&mut rng, 0u64..(1u64 << 20))).collect();

    let std_a: Vec<StdFr2> = bigs_a.iter().map(|f| StdFr2(f.into_bigint().0)).collect();
    let std_b: Vec<StdFr2> = bigs_b.iter().map(|f| StdFr2(f.into_bigint().0)).collect();

    // Correctness first: StdFr2::mul_bb must match arkworks' Fr*Fr exactly.
    for i in 0..1000 {
        let expected = (bigs_a[i] * bigs_b[i]).into_bigint().0;
        assert_eq!(std_a[i].mul_bb(&std_b[i]).0, expected, "mul_bb mismatch");
    }
    println!("[OK] StdFr2::mul_bb matches arkworks Fr*Fr on 1,000 random pairs.");

    let start = Instant::now();
    let mut sink_ark = Fr::ZERO;
    for i in 0..size { sink_ark += bigs_a[i] * bigs_b[i]; }
    sink_ark = std::hint::black_box(sink_ark);
    let t_arkworks = start.elapsed().as_secs_f64() * 1e9 / size as f64;

    let start = Instant::now();
    let mut sink_bb = StdFr2::zero();
    for i in 0..size { sink_bb = sink_bb.add(&std_a[i].mul_bb(&std_b[i])); }
    sink_bb = std::hint::black_box(sink_bb);
    let t_mul_bb = start.elapsed().as_secs_f64() * 1e9 / size as f64;

    let start = Instant::now();
    let mut sink_sb = StdFr2::zero();
    for i in 0..size { sink_sb = sink_sb.add(&std_a[i].mul_sb(smalls[i])); }
    sink_sb = std::hint::black_box(sink_sb);
    let t_mul_sb = start.elapsed().as_secs_f64() * 1e9 / size as f64;

    println!("--- Timing (ns/op, {size} ops) ---");
    println!("arkworks Fr*Fr (Montgomery, asm)     : {t_arkworks:8.2} ns");
    println!("StdFr2::mul_bb (Barrett, standard)   : {t_mul_bb:8.2} ns");
    println!("StdFr2::mul_sb (small-big, standard) : {t_mul_sb:8.2} ns");
    println!(
        "\nratio mul_bb / arkworks Fr*Fr : {:.3}x  (>1 => our big-big is slower)",
        t_mul_bb / t_arkworks
    );
    println!("ratio mul_sb / arkworks Fr*Fr : {:.3}x", t_mul_sb / t_arkworks);

    // assert! (not debug_assert!) -- keeps sink alive in --release, see the earlier
    // debug_assert! post-mortem in this same file's history.
    assert!(sink_ark != Fr::ZERO, "sink_ark optimized away");
    assert!(sink_bb != StdFr2::zero(), "sink_bb optimized away");
    assert!(sink_sb != StdFr2::zero(), "sink_sb optimized away");

    let filename = "csv/mul_bb_vs_arkworks.csv";
    let mut file = File::create(filename).expect("Unable to create mul_bb vs arkworks file");
    writeln!(file, "Operation,Time_ns").unwrap();
    writeln!(file, "arkworks Fr*Fr (Montgomery),{:.4}", t_arkworks).unwrap();
    writeln!(file, "StdFr2::mul_bb (Barrett),{:.4}", t_mul_bb).unwrap();
    writeln!(file, "StdFr2::mul_sb (small-big),{:.4}", t_mul_sb).unwrap();
    file.flush().unwrap();

    println!("\n[MUL_BB VS ARKWORKS OK] results written to csv/mul_bb_vs_arkworks.csv");
}

// =================================================================================================
// NEW ! TO UNDERSTAND
// 8. RAW_BARRETT_REDUCE ALGORITHMIC VARIANTS: mul_bb vs mul_bb_truncated vs mul_bb_mu4shift.
//    Two attempts at reducing the operation count of raw_barrett_reduce (see
//    bigint_field.rs's doc comments on StdFr2::mul_bb_truncated / mul_bb_mu4shift),
//    both correctness-verified but both measured SLOWER than the baseline once
//    averaged over multiple runs (single-shot ns-level timing was noisy enough to
//    flip sign between runs, hence the averaging here). Kept as a benchmark, not
//    silently dropped, because a well-verified negative result is itself evidence of
//    engineering work -- see the report's discussion of this experiment.
// =================================================================================================

pub fn bench_barrett_variants() {
    println!("==================================================");
    println!("  RAW_BARRETT_REDUCE VARIANTS: mul_bb vs truncated vs mu4shift  ");
    println!("==================================================");

    let mut rng = rand::thread_rng();

    // Correction d'abord, gros volume, contre arkworks comme verite.
    let n = 300_000;
    let mut mismatches_trunc = 0u64;
    let mut mismatches_mu4 = 0u64;
    for _ in 0..n {
        let a = Fr::rand(&mut rng);
        let b = Fr::rand(&mut rng);
        let sa = StdFr2(a.into_bigint().0);
        let sb = StdFr2(b.into_bigint().0);
        let expected = (a * b).into_bigint().0;
        if sa.mul_bb_truncated(&sb).0 != expected { mismatches_trunc += 1; }
        if sa.mul_bb_mu4shift(&sb).0 != expected { mismatches_mu4 += 1; }
    }
    println!("[Correction] mul_bb_truncated : {mismatches_trunc}/{n} erreurs.");
    println!("[Correction] mul_bb_mu4shift  : {mismatches_mu4}/{n} erreurs.");
    assert_eq!(mismatches_trunc, 0);
    assert_eq!(mismatches_mu4, 0);

    // Timing, moyenne sur plusieurs runs (le bruit de mesure change le signe du
    // resultat d'un run a l'autre a cette echelle -- voir la note ci-dessus).
    let size = 1_000_000;
    const NUM_TIMING_RUNS: usize = 7;
    let mut totals = [0.0f64; 4]; // arkworks Fr*Fr, mul_bb, mul_bb_truncated, mul_bb_mu4shift

    for _ in 0..NUM_TIMING_RUNS {
        let bigs_a: Vec<Fr> = (0..size).map(|_| Fr::rand(&mut rng)).collect();
        let bigs_b: Vec<Fr> = (0..size).map(|_| Fr::rand(&mut rng)).collect();
        let std_a: Vec<StdFr2> = bigs_a.iter().map(|f| StdFr2(f.into_bigint().0)).collect();
        let std_b: Vec<StdFr2> = bigs_b.iter().map(|f| StdFr2(f.into_bigint().0)).collect();

        let start = Instant::now();
        let mut sink_ark = Fr::ZERO;
        for i in 0..size { sink_ark += bigs_a[i] * bigs_b[i]; }
        sink_ark = std::hint::black_box(sink_ark);
        totals[0] += start.elapsed().as_secs_f64() * 1e9 / size as f64;
        assert!(sink_ark != Fr::ZERO);

        let start = Instant::now();
        let mut sink_bb = StdFr2::zero();
        for i in 0..size { sink_bb = sink_bb.add(&std_a[i].mul_bb(&std_b[i])); }
        sink_bb = std::hint::black_box(sink_bb);
        totals[1] += start.elapsed().as_secs_f64() * 1e9 / size as f64;
        assert!(sink_bb != StdFr2::zero());

        let start = Instant::now();
        let mut sink_trunc = StdFr2::zero();
        for i in 0..size { sink_trunc = sink_trunc.add(&std_a[i].mul_bb_truncated(&std_b[i])); }
        sink_trunc = std::hint::black_box(sink_trunc);
        totals[2] += start.elapsed().as_secs_f64() * 1e9 / size as f64;
        assert!(sink_trunc != StdFr2::zero());

        let start = Instant::now();
        let mut sink_mu4 = StdFr2::zero();
        for i in 0..size { sink_mu4 = sink_mu4.add(&std_a[i].mul_bb_mu4shift(&std_b[i])); }
        sink_mu4 = std::hint::black_box(sink_mu4);
        totals[3] += start.elapsed().as_secs_f64() * 1e9 / size as f64;
        assert!(sink_mu4 != StdFr2::zero());
    }

    let avg: Vec<f64> = totals.iter().map(|t| t / NUM_TIMING_RUNS as f64).collect();

    println!("--- Moyenne sur {NUM_TIMING_RUNS} runs de {size} operations ---");
    println!("arkworks Fr*Fr                   : {:8.2} ns", avg[0]);
    println!("StdFr2::mul_bb (baseline)         : {:8.2} ns", avg[1]);
    println!("StdFr2::mul_bb_truncated          : {:8.2} ns   ({:+.1}% vs baseline)", avg[2], 100.0*(1.0 - avg[2]/avg[1]));
    println!("StdFr2::mul_bb_mu4shift           : {:8.2} ns   ({:+.1}% vs baseline)", avg[3], 100.0*(1.0 - avg[3]/avg[1]));

    let filename = "csv/barrett_variants.csv";
    let mut file = File::create(filename).expect("Unable to create barrett variants file");
    writeln!(file, "Operation,Time_ns").unwrap();
    writeln!(file, "arkworks Fr*Fr,{:.4}", avg[0]).unwrap();
    writeln!(file, "StdFr2::mul_bb (baseline),{:.4}", avg[1]).unwrap();
    writeln!(file, "StdFr2::mul_bb_truncated,{:.4}", avg[2]).unwrap();
    writeln!(file, "StdFr2::mul_bb_mu4shift,{:.4}", avg[3]).unwrap();
    file.flush().unwrap();

    println!("\n[BARRETT VARIANTS OK] results written to csv/barrett_variants.csv");
}