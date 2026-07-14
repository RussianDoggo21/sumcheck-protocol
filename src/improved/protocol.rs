use crate::improved::engine::EvaluationPoint;
use crate::improved::engine::{
    fold_hypercube_chunk, get_u_hat_domain, dynamic_folding_step, multi_product_eval,
    precompute_kernel_cache, compute_lagrange_denominator_invs,
};
use crate::improved::arithmetic::{FAST_PATH_COUNT, SLOW_PATH_COUNT};
use crate::improved::prover::Prover;
use crate::improved::streaming::PolynomialStream;
use crate::improved::verifier::Verifier;
use ark_ff::{Field, PrimeField};
use ark_poly::DenseMultilinearExtension;
use ark_test_curves::bls12_381::Fr;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use std::sync::atomic::Ordering;

pub trait SumcheckProtocol<F: PrimeField> {
    fn run(&self, stream: &mut dyn PolynomialStream<F>, sumcheck_claim: F) -> bool;
    fn run_sb_1(&self, stream: &mut dyn PolynomialStream<F>, sumcheck_claim: F) -> bool;
}

pub struct LinearTimeSC;

impl SumcheckProtocol<Fr> for LinearTimeSC {
    fn run(&self, stream: &mut dyn PolynomialStream<Fr>, sumcheck_claim: Fr) -> bool {
        let num_vars = stream.num_vars();
        let num_poly = stream.degree();
        stream.rewind();

        let full_chunk = stream
            .next_chunk(1 << num_vars)
            .expect("Stream should provide the full hypercube data");

        let list_of_poly: Vec<DenseMultilinearExtension<Fr>> = full_chunk
            .into_iter()
            .map(|evals| DenseMultilinearExtension::from_evaluations_vec(num_vars, evals))
            .collect();

        Self::linear_time_sc(&list_of_poly, num_poly, sumcheck_claim)
    }

    fn run_sb_1(&self, stream: &mut dyn PolynomialStream<Fr>, sumcheck_claim: Fr) -> bool {
        let num_vars = stream.num_vars();
        let num_poly = stream.degree();
        stream.rewind();

        let full_chunk = stream
            .next_chunk(1 << num_vars)
            .expect("Stream should provide the full hypercube data");

        let list_of_poly: Vec<DenseMultilinearExtension<Fr>> = full_chunk
            .into_iter()
            .map(|evals| DenseMultilinearExtension::from_evaluations_vec(num_vars, evals))
            .collect();

        Self::linear_time_sc_sb_1(&list_of_poly, num_poly, sumcheck_claim)
    }
}

impl LinearTimeSC {
    pub fn linear_time_sc(
        list_of_poly: &[DenseMultilinearExtension<Fr>],
        num_poly: usize,
        sumcheck_claim: Fr,
    ) -> bool {
        assert!(list_of_poly.len() > 0, "Cannot run sumcheck on an empty list of polynomials");
        let num_vars = list_of_poly[0].num_vars;
        for p in list_of_poly {
            assert_eq!(num_vars, p.num_vars, "All polynomials must have the same number of variables");
        }
        assert_eq!(list_of_poly.len(), num_poly, "Number of multilinear polynomials must equal {num_poly}");

        let mut prover = Prover::new(list_of_poly);
        let mut verifier = Verifier::new(num_poly);
        let mut c_i = sumcheck_claim;
        let mut rng = rand::thread_rng();

        for i in 0..num_vars {
            let s_i = prover.compute_s_i(num_vars, i);
            verifier.add_s_i(s_i);
            let s_i_0 = verifier.compute_s_i_0(c_i);
            let challenge = verifier.send_challenge(&mut rng);
            c_i = verifier.update_c_i(challenge, s_i_0);
            prover.update_p_arrays(num_vars, i, challenge);
        }

        verifier.final_check(list_of_poly, c_i)
    }

    pub fn linear_time_sc_sb_1(
        list_of_poly: &[DenseMultilinearExtension<Fr>],
        num_poly: usize,
        sumcheck_claim: Fr,
    ) -> bool {
        assert!(list_of_poly.len() > 0, "Cannot run sumcheck on an empty list of polynomials");
        let num_vars = list_of_poly[0].num_vars;
        for p in list_of_poly {
            assert_eq!(num_vars, p.num_vars, "All polynomials must have the same number of variables");
        }
        assert_eq!(list_of_poly.len(), num_poly, "Number of multilinear polynomials must equal {num_poly}");

        let mut prover = Prover::new(list_of_poly);
        let mut verifier = Verifier::new(num_poly);
        let mut c_i = sumcheck_claim;
        let mut rng = rand::thread_rng();

        let s_0 = prover.compute_s_0_sb(num_vars);
        verifier.add_s_i(s_0);
        let s_0_0 = verifier.compute_s_i_0(c_i);
        let challenge = verifier.send_challenge(&mut rng);
        c_i = verifier.update_c_i(challenge, s_0_0);
        prover.update_p_arrays(num_vars, 0, challenge);

        for i in 1..num_vars {
            let s_i = prover.compute_s_i(num_vars, i);
            verifier.add_s_i(s_i);
            let s_i_0 = verifier.compute_s_i_0(c_i);
            let challenge = verifier.send_challenge(&mut rng);
            c_i = verifier.update_c_i(challenge, s_i_0);
            prover.update_p_arrays(num_vars, i, challenge);
        }

        verifier.final_check(list_of_poly, c_i)
    }
}

/// The Small-Value (SV) optimized Sumcheck protocol.
///
/// NEW ! TO UNDERSTAND : per NAIST review, EvalProductSV does not actually expose a real
/// offline/online split to its callers -- `run` below is now the single entry point. The
/// grid-construction / interactive-rounds separation still exists internally (as private
/// helpers) purely to let `run_sequential` share the interactive-rounds code with `run`
/// without duplicating it, and to keep the chunk-level parallelism benchmarkable
/// (see `bench_run_seq_vs_parallel` in benchmark.rs). Neither `precomputation_phase` nor
/// `online_phase` are part of the public API anymore.
pub struct EvalProductSV {
    pub early_window_size: usize,
}

impl EvalProductSV {
    pub fn new(d: usize, num_vars: usize) -> Self {
        let kappa = 78.0;
        let d_f = d as f64;
        let argument = d_f * d_f * kappa;
        let base = d_f + 1.0;
        let v_star_exact = argument.ln() / base.ln();

        let mut omega_1 = v_star_exact.round() as usize;
        if omega_1 < 1 { omega_1 = 1; }
        if omega_1 > num_vars { omega_1 = num_vars; }

        Self { early_window_size: omega_1 }
    }

    /// Builds the extrapolated evaluation grid over the early window, processing chunks in
    /// parallel (rayon). Private: not a separately callable "offline phase" anymore.
    fn build_grid_parallel(&self, stream: &dyn PolynomialStream<Fr>) -> Vec<Fr> {
        let d = stream.degree();
        let early_window_size = self.early_window_size;
        let grid_size = usize::pow(d + 1, early_window_size as u32);
        let chunk_size = 1 << early_window_size;
        let total_size = 1usize << stream.num_vars();
        let num_chunks = total_size / chunk_size;

        // NEW ! TO UNDERSTAND : precomputed ONCE and shared read-only across every chunk /
        // rayon worker thread, instead of `multivariate_extrapolate` silently recomputing
        // `compute_kernel(k)` (k modular inversions) from scratch on every chunk.
        let kernel_cache = precompute_kernel_cache(d);

        // NEW ! TO UNDERSTAND : forces a coarser grain than rayon's default adaptive splitting.
        // Without this, for large num_chunks (e.g. num_vars=14 with a small early_window_size),
        // rayon ends up with far more tasks than useful work per task, causing constant
        // sleep/wake churn between workers (perf showed ~20%+ cumulative in `schedule`,
        // `finish_task_switch`, `futex_wait`, `Sleep::sleep`). Targeting ~4 tasks per worker
        // thread keeps load-balancing reasonable while sharply cutting the number of
        // steal/wake events.
        let min_len = (num_chunks / (rayon::current_num_threads() * 4)).max(1);

        (0..num_chunks)
            .into_par_iter()
            .with_min_len(min_len)
            .map(|chunk_idx| {
                let chunk = stream.chunk_at(chunk_idx, chunk_size);
                // NEW ! TO UNDERSTAND : local to this chunk/task, no atomics involved while
                // multi_product_eval recurses -- flushed into the real global atomics ONCE
                // below, instead of once per term (see the note in arithmetic.rs).
                let mut local_fast: u64 = 0;
                let mut local_slow: u64 = 0;
                let partial = multi_product_eval(&chunk, d, early_window_size, &kernel_cache, &mut local_fast, &mut local_slow);
                if local_fast > 0 {
                    FAST_PATH_COUNT.fetch_add(local_fast, Ordering::Relaxed);
                }
                if local_slow > 0 {
                    SLOW_PATH_COUNT.fetch_add(local_slow, Ordering::Relaxed);
                }
                partial
            })
            .reduce(
                || vec![Fr::ZERO; grid_size],
                |mut acc, partial| {
                    for i in 0..grid_size {
                        acc[i] += partial[i];
                    }
                    acc
                },
            )
    }

    /// NEW ! TO UNDERSTAND : sequential twin of `build_grid_parallel`, kept ONLY so
    /// `run_sequential` (below) can be benchmarked against `run` to measure the impact of the
    /// chunk-level parallelism on the full protocol. Never called by `run` itself.
    fn build_grid_sequential(&self, stream: &mut dyn PolynomialStream<Fr>) -> Vec<Fr> {
        let d = stream.degree();
        let early_window_size = self.early_window_size;
        let grid_size = usize::pow(d + 1, early_window_size as u32);
        let chunk_size = 1 << early_window_size;
        let total_size = 1usize << stream.num_vars();
        let num_chunks = total_size / chunk_size;

        // NEW ! TO UNDERSTAND : same fix as build_grid_parallel -- precompute once, reuse
        // across every chunk instead of recomputing compute_kernel(k) per chunk.
        let kernel_cache = precompute_kernel_cache(d);

        let mut q = vec![Fr::ZERO; grid_size];
        for chunk_idx in 0..num_chunks {
            let chunk = stream.chunk_at(chunk_idx, chunk_size);
            let mut local_fast: u64 = 0;
            let mut local_slow: u64 = 0;
            let partial = multi_product_eval(&chunk, d, early_window_size, &kernel_cache, &mut local_fast, &mut local_slow);
            if local_fast > 0 {
                FAST_PATH_COUNT.fetch_add(local_fast, Ordering::Relaxed);
            }
            if local_slow > 0 {
                SLOW_PATH_COUNT.fetch_add(local_slow, Ordering::Relaxed);
            }
            for i in 0..grid_size {
                q[i] += partial[i];
            }
        }
        q
    }

    /// The whole EvalProductSV protocol (interactive rounds over the early window, then the
    /// linear-time transition and final oracle check), as a single block -- this is the code
    /// that used to be split across `precomputation_phase` and `online_phase`. Shared by both
    /// `run` and `run_sequential`; the only difference between them is how `q` was built.
    fn run_with_grid(&self, stream: &mut dyn PolynomialStream<Fr>, sumcheck_claim: Fr, mut q: Vec<Fr>) -> bool {
        let num_vars = stream.num_vars();
        let d = stream.degree();
        let early_window_size = self.early_window_size;
        let chunk_size = 1 << early_window_size;

        let mut verifier = Verifier::new(d);
        let mut c_i = sumcheck_claim;
        let mut rng = rand::thread_rng();

        let u_d_hat = get_u_hat_domain(d);
        let base = d + 1;

        // NEW ! TO UNDERSTAND : precomputed once per protocol run (d is fixed throughout) --
        // see compute_lagrange_denominator_invs. Previously dynamic_folding_step recomputed
        // these `d` modular inversions from scratch on every round of the early window.
        let denom_inv = compute_lagrange_denominator_invs::<Fr>(d);

        for i in 0..early_window_size {
            let mut s_i_evals = vec![Fr::ZERO; d];
            let remaining_vars = early_window_size - 1 - i;
            let next_hypercube_size = 1 << remaining_vars;

            for (u_idx, &u) in u_d_hat.iter().enumerate() {
                let mut sum_over_hypercube = Fr::ZERO;
                let current_var_idx = match u {
                    EvaluationPoint::Infinity => 0,
                    EvaluationPoint::Value(v) => (v + 1) as usize,
                };

                for x_prime in 0..next_hypercube_size {
                    let mut future_offset = 0;
                    for var_j in 0..remaining_vars {
                        let bit = (x_prime >> var_j) & 1;
                        let grid_val_idx = if bit == 0 { 1 } else { 2 };
                        future_offset += grid_val_idx * usize::pow(base, (var_j + 1) as u32);
                    }
                    let final_grid_index = current_var_idx + future_offset;
                    sum_over_hypercube += q[final_grid_index];
                }
                s_i_evals[u_idx] = sum_over_hypercube;
            }

            verifier.add_s_i(s_i_evals);
            let challenge = verifier.send_challenge(&mut rng);
            let s_i_0 = verifier.compute_s_i_0(c_i);
            c_i = verifier.update_c_i(challenge, s_i_0);

            if remaining_vars > 0 {
                q = dynamic_folding_step(&q, challenge, d, base, remaining_vars, &denom_inv);
            }
        }

        let remaining_rounds = num_vars - early_window_size;
        let final_hypercube_size = 1 << remaining_rounds;
        let mut final_prover_arrays = vec![vec![Fr::ZERO; final_hypercube_size]; d];
        stream.rewind();

        let mut element_idx = 0;
        while let Some(chunk) = stream.next_chunk(chunk_size) {
            for k in 0..d {
                let folded_scalar = fold_hypercube_chunk(&chunk[k], &verifier.challenges[0..early_window_size]);
                final_prover_arrays[k][element_idx] = folded_scalar;
            }
            element_idx += 1;
        }

        let mut prover = Prover::with_arrays(final_prover_arrays);
        for i in early_window_size..num_vars {
            let s_i = prover.compute_s_i(num_vars, i);
            verifier.add_s_i(s_i);
            let s_i_0 = verifier.compute_s_i_0(c_i);
            let challenge = verifier.send_challenge(&mut rng);
            c_i = verifier.update_c_i(challenge, s_i_0);
            prover.update_p_arrays(num_vars, i, challenge);
        }

        let final_evals = stream.evaluate_at_point(&verifier.challenges);
        let mut expected_product = Fr::ONE;
        for val in final_evals {
            expected_product *= val;
        }

        c_i == expected_product
    }

    /// NEW ! TO UNDERSTAND : sequential entry point (grid built without rayon). Behaviorally
    /// identical to `run` -- used only by `bench_run_seq_vs_parallel` in benchmark.rs.
    pub fn run_sequential(&self, stream: &mut dyn PolynomialStream<Fr>, sumcheck_claim: Fr) -> bool {
        let q = self.build_grid_sequential(stream);
        self.run_with_grid(stream, sumcheck_claim, q)
    }
}

impl SumcheckProtocol<Fr> for EvalProductSV {
    fn run(&self, stream: &mut dyn PolynomialStream<Fr>, sumcheck_claim: Fr) -> bool {
        let q = {
            let stream_ref: &dyn PolynomialStream<Fr> = &*stream;
            self.build_grid_parallel(stream_ref)
        };
        self.run_with_grid(stream, sumcheck_claim, q)
    }

    // FAKE
    fn run_sb_1(&self, _stream: &mut dyn PolynomialStream<Fr>, _sumcheck_claim: Fr) -> bool {
        true
    }
}