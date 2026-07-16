// Benchmark for the d=1 (single multilinear polynomial) case.

use std::time::{Duration, Instant};
use std::fs::File;
use std::io::Write;

use ark_poly::DenseMultilinearExtension;
use ark_ff::PrimeField;
use ark_linear_sumcheck::ml_sumcheck::MLSumcheck;
use ark_test_curves::bls12_381::Fr;

use crate::naive::protocol::sc_protocol as sc_protocol_naive;
use crate::utils::{compute_hypercube_sum, generate_poly_test, generate_small_evaluations_from_poly};
use crate::improved::protocol::{LinearTimeSC, SumcheckProtocol};
use crate::improved::streaming::MockStream;

const NUM_RUNS: u32 = 5;

fn run_optimized_case(num_vars: usize, evals: Vec<Fr>, sumcheck_claim: Fr) -> bool {
    let poly = DenseMultilinearExtension::from_evaluations_vec(num_vars, evals);
    let list_of_poly = vec![poly];
    let mut stream = MockStream::new(num_vars, 1, &list_of_poly);
    let prover = LinearTimeSC;
    prover.run(&mut stream, sumcheck_claim)
}

pub fn bench_naive_vs_arkworks_vs_optimized() {
    println!("==================================================");
    println!("  D=1 SANITY CHECK: naive vs arkworks vs optimized (LinearTimeSC)  ");
    println!("==================================================");

    let filename = "csv/naive_vs_arkworks_vs_optimized.csv";
    let mut file = File::create(filename).expect("Unable to create naive vs arkworks vs optimized file");
    writeln!(file, "Run,NumVars,Naive_ms,Arkworks_ms,Optimized_ms,Naive_Sum,Arkworks_Sum,Optimized_Sum").unwrap();

    let mut total_naive = Duration::ZERO;
    let mut total_arkworks = Duration::ZERO;
    let mut total_optimized = Duration::ZERO;

    for run in 1..=NUM_RUNS {
        println!("\n--- Run {run}/{NUM_RUNS} ---");
        let mut rng = rand::thread_rng();

        let (poly0, list_of_products) = generate_poly_test(&mut rng);
        let num_vars = poly0.num_vars;
        println!("   num_vars = {num_vars}");

        // -------------------- Naive --------------------
        let gamma = compute_hypercube_sum(&poly0);
        let start = Instant::now();
        let ok_naive = sc_protocol_naive(&poly0, gamma, &mut rng);
        let t_naive = start.elapsed();
        assert!(ok_naive, "naive protocol REJECTED on run {run}");
        total_naive += t_naive;
        println!("   naive     : {:8.3} ms (sum = {gamma})", t_naive.as_secs_f64() * 1000.0);

        // -------------------- Arkworks --------------------
        let start = Instant::now();
        let proof = MLSumcheck::prove(&list_of_products).expect("The arkworks prover failed");
        let t_arkworks = start.elapsed();
        let claimed_sum = MLSumcheck::extract_sum(&proof);
        total_arkworks += t_arkworks;
        println!("   arkworks  : {:8.3} ms (sum = {})", t_arkworks.as_secs_f64() * 1000.0, claimed_sum.into_bigint());

        // -------------------- Optimized (current LinearTimeSC, d=1) --------------------
        // NEW ! TO UNDERSTAND : generate_small_evaluations_from_poly returns Vec<u64>
        // (the raw small-value representation used elsewhere in the small-value
        // machinery), but run_optimized_case / DenseMultilinearExtension need Vec<Fr>
        // -- convert element-wise via Fr::from before handing off.
        let evals_small_u64: Vec<u64> = generate_small_evaluations_from_poly(&poly0);
        let evals_small: Vec<Fr> = evals_small_u64.into_iter().map(Fr::from).collect();

        let start = Instant::now();
        let ok_optimized = run_optimized_case(num_vars, evals_small, gamma);
        let t_optimized = start.elapsed();
        assert!(ok_optimized, "optimized (LinearTimeSC) protocol REJECTED on run {run}");
        total_optimized += t_optimized;
        println!("   optimized : {:8.3} ms", t_optimized.as_secs_f64() * 1000.0);

        writeln!(
            file,
            "{},{},{:.4},{:.4},{:.4},{},{},{}",
            run,
            num_vars,
            t_naive.as_secs_f64() * 1000.0,
            t_arkworks.as_secs_f64() * 1000.0,
            t_optimized.as_secs_f64() * 1000.0,
            gamma,
            claimed_sum.into_bigint(),
            "accepted"
        ).unwrap();
        file.flush().unwrap();
    }

    let avg_naive_ms = (total_naive / NUM_RUNS).as_secs_f64() * 1000.0;
    let avg_arkworks_ms = (total_arkworks / NUM_RUNS).as_secs_f64() * 1000.0;
    let avg_optimized_ms = (total_optimized / NUM_RUNS).as_secs_f64() * 1000.0;

    println!("\n--- Averages over {NUM_RUNS} runs ---");
    println!("naive     : {avg_naive_ms:8.3} ms");
    println!("arkworks  : {avg_arkworks_ms:8.3} ms");
    println!("optimized : {avg_optimized_ms:8.3} ms");
    println!(
        "\nspeedup optimized vs naive     : {:.2}x",
        avg_naive_ms / avg_optimized_ms
    );
    println!(
        "speedup optimized vs arkworks  : {:.2}x",
        avg_arkworks_ms / avg_optimized_ms
    );

    println!("\n[D=1 SANITY CHECK OK] results written to csv/naive_vs_arkworks_vs_optimized.csv");
}