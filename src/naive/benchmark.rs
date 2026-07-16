// Benchmark for the d=1 (single multilinear polynomial) case: this was the very
// first working milestone of the project, before the multivariate (product of d
// polynomials) pipeline existed. Kept and benchmarked separately from
// improved/benchmark.rs because it exercises a genuinely different code path (the
// naive module operates on ark_poly's SparsePolynomial/SparseTerm representation,
// not the DenseMultilinearExtension-based streaming pipeline of the current
// `improved` module).
//
// The "optimized" comparison point below is the project's own historical
// `sc_protocol_improved` / `fast_dot_product` routines (reinstated here verbatim,
// with `small_big_mul` renamed to the current `small_big_mul_raw`, which has the
// exact same `(small: u64, big: &Fr) -> Fr` signature) -- NOT a stand-in from the
// current `improved` architecture.

use std::time::{Duration, Instant};
use std::fs::File;
use std::io::Write;

use ark_ff::PrimeField;
use ark_linear_sumcheck::ml_sumcheck::MLSumcheck;
use ark_test_curves::bls12_381::Fr;

use crate::naive::protocol::sc_protocol as sc_protocol_naive;
use crate::utils::{compute_hypercube_sum, generate_poly_test, generate_small_evaluations_from_poly};
use crate::improved::arithmetic::small_big_mul_raw;

const NUM_RUNS: u32 = 5;

// ================================================================================
// Historical "Small-Value" optimized prover for the d=1 case (reinstated verbatim
// from the project's early prototype, see the report's Section on this milestone).
// ================================================================================

/// Fast dot product combining lazy reduction and small-big multiplication.
/// Reinstated as-is; `small_big_mul` renamed to `small_big_mul_raw` (same signature,
/// current name for the same routine in `improved::arithmetic`).
fn fast_dot_product(small_values: &[u64], coefficients: &[Fr]) -> Fr {
    assert_eq!(small_values.len(), coefficients.len(), "Taille incoherente");

    let mut final_res = Fr::from(0u64);
    let mut accumulator_u128: u128 = 0;

    for (&small, coeff) in small_values.iter().zip(coefficients.iter()) {
        if *coeff == Fr::from(1u64) {
            accumulator_u128 += small as u128;
        } else if *coeff == Fr::from(0u64) {
            continue;
        } else {
            if accumulator_u128 > 0 {
                final_res += Fr::from(accumulator_u128);
                accumulator_u128 = 0;
            }
            final_res += small_big_mul_raw(small, coeff);
        }
    }

    if accumulator_u128 > 0 {
        final_res += Fr::from(accumulator_u128);
    }

    final_res
}

/// Optimized "Small-Value" Sumcheck protocol implementation for a single multilinear
/// polynomial (d=1 case), reinstated verbatim from the project's original prototype.
/// Round 0 stays entirely in raw u64/small-value form; from round 1 onward,
/// evaluations are upgraded to full Fr elements. Uses a FIXED, deterministic
/// challenge schedule internally (r_i = i+2), not real randomness -- a known
/// limitation of this historical prototype, not something changed here.
fn sc_protocol_improved(num_vars: usize, small_evals: &[u64]) -> (Fr, Vec<(Fr, Fr)>) {
    let mut proofs = Vec::with_capacity(num_vars);

    let mut total_sum_u128: u128 = 0;
    for &val in small_evals {
        total_sum_u128 += val as u128;
    }
    let claimed_sum = Fr::from(total_sum_u128);

    let mut challenges = Vec::with_capacity(num_vars);
    for i in 0..num_vars {
        challenges.push(Fr::from((i + 2) as u64));
    }

    // --- ROUND 0: Pure Small-Value Optimization ---
    let half = small_evals.len() / 2;
    let mut p0_small = Vec::with_capacity(half);
    let mut p1_small = Vec::with_capacity(half);

    for i in 0..half {
        p0_small.push(small_evals[i * 2]);
        p1_small.push(small_evals[i * 2 + 1]);
    }

    let coeffs = vec![Fr::from(1u64); half];
    let p0_r0 = fast_dot_product(&p0_small, &coeffs);
    let p1_r0 = fast_dot_product(&p1_small, &coeffs);
    proofs.push((p0_r0, p1_r0));

    let r0 = challenges[0];
    let mut current_fr_evals = Vec::with_capacity(half);
    for i in 0..half {
        let p0_fr = Fr::from(p0_small[i]);
        let p1_fr = Fr::from(p1_small[i]);
        let next_val = p0_fr + r0 * (p1_fr - p0_fr);
        current_fr_evals.push(next_val);
    }

    // --- ROUNDS 1 to num_vars-1: Standard/Hybrid Proving ---
    let mut current_size = half;

    for round in 1..num_vars {
        let next_half = current_size / 2;
        let mut p0_fr_vec = Vec::with_capacity(next_half);
        let mut p1_fr_vec = Vec::with_capacity(next_half);

        for i in 0..next_half {
            p0_fr_vec.push(current_fr_evals[i * 2]);
            p1_fr_vec.push(current_fr_evals[i * 2 + 1]);
        }

        let mut p0 = Fr::from(0u64);
        let mut p1 = Fr::from(0u64);
        for i in 0..next_half {
            p0 += p0_fr_vec[i];
            p1 += p1_fr_vec[i];
        }
        proofs.push((p0, p1));

        let r = challenges[round];
        let mut next_fr_evals = Vec::with_capacity(next_half);
        for i in 0..next_half {
            let next_val = p0_fr_vec[i] + r * (p1_fr_vec[i] - p0_fr_vec[i]);
            next_fr_evals.push(next_val);
        }

        current_fr_evals = next_fr_evals;
        current_size = next_half;
    }

    (claimed_sum, proofs)
}

/// NEW ! TO UNDERSTAND : `sc_protocol_improved` returns the prover's transcript
/// (claimed_sum, per-round (p0,p1) pairs) but performs no verification itself -- it
/// is a prover-only routine. To get a fair, apples-to-apples "accepted/rejected"
/// comparison against the naive and arkworks provers (both of which run an actual
/// interactive check), this wrapper replays the verifier's side of the protocol
/// using the SAME fixed challenge schedule the prover used internally (r_i = i+2):
/// checks that claimed_sum matches the external ground truth, and that each round's
/// (p0, p1) satisfies p0 + p1 == the running claim, folding the claim via
/// p0 + r_i*(p1-p0) exactly as the standard sum-check relation requires.
fn run_sc_protocol_improved_checked(num_vars: usize, small_evals: &[u64], expected_sum: Fr) -> bool {
    let (claimed_sum, proofs) = sc_protocol_improved(num_vars, small_evals);

    if claimed_sum != expected_sum {
        return false;
    }

    let mut current_claim = claimed_sum;
    for (round, &(p0, p1)) in proofs.iter().enumerate() {
        if p0 + p1 != current_claim {
            return false;
        }
        let r = Fr::from((round + 2) as u64);
        current_claim = p0 + r * (p1 - p0);
    }

    true
}

pub fn bench_naive_vs_arkworks_vs_optimized() {
    println!("==================================================");
    println!("  D=1 SANITY CHECK: naive vs arkworks vs optimized (sc_protocol_improved)  ");
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

        // -------------------- Optimized (historical sc_protocol_improved) --------------------
        let evals_small: Vec<u64> = generate_small_evaluations_from_poly(&poly0);
        let start = Instant::now();
        let ok_optimized = run_sc_protocol_improved_checked(num_vars, &evals_small, gamma);
        let t_optimized = start.elapsed();
        assert!(ok_optimized, "optimized (sc_protocol_improved) protocol REJECTED on run {run}");
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