// First implementation of sumcheck protocol using arkworks

mod improved;
mod utils;

use ark_ff::Field;
use ark_linear_sumcheck::ml_sumcheck::MLSumcheck;
use ark_poly::MultilinearExtension;
use ark_test_curves::bls12_381::Fr;

use std::fs::File;
use std::io::{Write, stdout};
use std::time::{Duration, Instant};

use crate::improved::protocol::{EvalProductSV, LinearTimeSC, SumcheckProtocol};
use crate::improved::streaming::MockStream;
use crate::utils::generate_multivariate_poly_test;

fn main() {
    // Parameters configuration:
    // - max_vars : Benchmark from 4 up to max_vars variables (2^max_vars points)
    // - num_runs : Number of iterations per variable count to get a stable average
    let max_vars = 14; // Réduit à 14 pour éviter des temps d'attente trop longs à haut degré (d=9)
    let num_runs = 5;

    // Array of degrees to analyze sequentially
    let degrees_to_test = [3];

    println!("==================================================");
    println!("       STARTING SUMCHECK PROTOCOL BENCHMARK        ");
    println!("==================================================");

    for &d in &degrees_to_test {
        println!("\n##################################################");
        println!("  LAUNCHING BENCHMARK SERIES FOR DEGREE d = {}", d);
        println!("##################################################");
        test_range_variables(max_vars, d, num_runs);
    }

    println!("\n[GLOBAL OK] All benchmarks completed successfully!");
}

/// Orchestrates the benchmarks over a range of variables for a given degree d
/// and saves the results into a distinct, degree-specific CSV file.
pub fn test_range_variables(max_vars: usize, d: usize, num_runs: u32) {
    // Dynamic filename based on the current degree parameter
    let filename = format!("benchmark_results_d{}.csv", d);
    let mut file = File::create(&filename).expect("Unable to create file");

    // Write CSV header to include EvalProductSV
    writeln!(
        file,
        "Variables,Arkworks_ms,LinearTimeSC_ms,EvalProductSV_ms"
    )
    .unwrap();
    file.flush().unwrap();

    // Warm up from 4 variables up to max_vars
    for num_v in 4..=max_vars {
        println!("\n==================================================");
        println!(
            " Benchmarking for {} variables (2^{} = {} points, d={})",
            num_v,
            num_v,
            1 << num_v,
            d
        );
        println!(" Average over {} runs...", num_runs);
        println!("==================================================");

        let mut total_ark = Duration::ZERO;
        let mut total_opt = Duration::ZERO;
        let mut total_sv = Duration::ZERO;

        for run in 1..=num_runs {
            print!("   Run {}/{}... ", run, num_runs);
            stdout().flush().unwrap();

            let (d_ark, d_opt, d_sv) = multivariate_test(num_v, d);

            total_ark += d_ark;
            total_opt += d_opt;
            total_sv += d_sv;

            println!("Done.");
        }

        // Compute averages
        let avg_ark = total_ark / num_runs;
        let avg_opt = total_opt / num_runs;
        let avg_sv = total_sv / num_runs;

        // Convert Durations to milliseconds (f64)
        let duration_ark_ms = avg_ark.as_secs_f64() * 1000.0;
        let duration_opt_ms = avg_opt.as_secs_f64() * 1000.0;
        let duration_sv_ms = avg_sv.as_secs_f64() * 1000.0;

        // Save benchmarks to the dynamic CSV file
        writeln!(
            file,
            "{},{},{},{}",
            num_v, duration_ark_ms, duration_opt_ms, duration_sv_ms
        )
        .unwrap();
        file.flush().unwrap();

        println!("\n -> Average Arkworks     : {:.4} ms", duration_ark_ms);
        println!(" -> Average LinearTimeSC  : {:.4} ms", duration_opt_ms);
        println!(" -> Average EvalProductSV : {:.4} ms", duration_sv_ms);
    }
    println!("\n[OK] Series complete! Results saved in '{}'.", filename);
}

/// Runs a single multivariate test instance comparing Arkworks, LinearTimeSC, and EvalProductSV.
fn multivariate_test(num_vars: usize, d: usize) -> (Duration, Duration, Duration) {
    let mut rng = rand::thread_rng();

    // 1. Generate the random multilinear extensions and the official Arkworks data structure
    let (list_of_poly, list_of_products) = generate_multivariate_poly_test(&mut rng, num_vars, d);

    // 2. Local exact sum calculation over the hypercube to provide the correct 'expected_sum' claim
    let hypercube_size = 1 << num_vars;
    let mut expected_sum = Fr::ZERO;
    for x in 0..hypercube_size {
        let mut product_at_x = Fr::ONE;
        for k in 0..d {
            product_at_x *= list_of_poly[k].evaluations[x];
        }
        expected_sum += product_at_x;
    }

    // 3. Benchmark Arkworks native multi-factor MLSumcheck implementation
    let start_arkworks = Instant::now();
    let proof = MLSumcheck::prove(&list_of_products).expect("The Arkworks prover failed");
    let duration_arkworks = start_arkworks.elapsed();

    // Verify Arkworks sum matches our calculated one (sanity check)
    let ark_sum = MLSumcheck::extract_sum(&proof);
    assert_eq!(
        expected_sum, ark_sum,
        "Local hypercube sum mismatch with Arkworks sum extraction!"
    );

    // 4. Benchmark our custom interactive LinearTime_SC protocol
    let linear_time_protocol = LinearTimeSC;
    let l = list_of_poly[0].num_vars();
    let d_len = list_of_poly.len();
    let mut stream_opt = MockStream::new(l, d_len, &list_of_poly);

    let start_improved = Instant::now();
    let verifier_accepted_opt = linear_time_protocol.run(&mut stream_opt, expected_sum);
    let duration_improved = start_improved.elapsed();

    assert!(
        verifier_accepted_opt,
        "Security Error: The LinearTimeSC Verifier REJECTED the proof for {} variables!",
        num_vars
    );

    // 5. Benchmark our custom EvalProductSV protocol (Small-Value / Grid Window emulation)
    let eval_product_sv_protocol = EvalProductSV::new(d_len, l);
    let mut stream_sv = MockStream::new(l, d_len, &list_of_poly);

    let start_sv = Instant::now();
    let verifier_accepted_sv = eval_product_sv_protocol.run(&mut stream_sv, expected_sum);
    let duration_sv = start_sv.elapsed();

    assert!(
        verifier_accepted_sv,
        "Security Error: The EvalProductSV Verifier REJECTED the proof for {} variables!",
        num_vars
    );

    (duration_arkworks, duration_improved, duration_sv)
}