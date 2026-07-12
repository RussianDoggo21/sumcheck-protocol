use crate::improved::engine::EvaluationPoint;
use crate::improved::engine::{
    fold_hypercube_chunk, get_u_hat_domain, dynamic_folding_step, multi_product_eval,
};
use crate::improved::prover::Prover;
use crate::improved::streaming::PolynomialStream;
use crate::improved::verifier::Verifier;
use ark_ff::{Field, PrimeField};
use ark_poly::DenseMultilinearExtension;
use ark_test_curves::bls12_381::Fr;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

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
        assert!(
            list_of_poly.len() > 0,
            "Cannot run sumcheck on an empty list of polynomials"
        );

        let num_vars = list_of_poly[0].num_vars;
        for p in list_of_poly {
            assert_eq!(
                num_vars, p.num_vars,
                "All polynomials must have the same number of variables"
            );
        }

        assert_eq!(
            list_of_poly.len(),
            num_poly,
            "Number of multilinear polynomials must equal {num_poly}"
        );

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
        assert!(
            list_of_poly.len() > 0,
            "Cannot run sumcheck on an empty list of polynomials"
        );

        let num_vars = list_of_poly[0].num_vars;
        for p in list_of_poly {
            assert_eq!(
                num_vars, p.num_vars,
                "All polynomials must have the same number of variables"
            );
        }

        assert_eq!(
            list_of_poly.len(),
            num_poly,
            "Number of multilinear polynomials must equal {num_poly}"
        );

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
pub struct EvalProductSV {
    pub early_window_size: usize,
}

/// Stores exclusively the offline data structure which DOES NOT depend on any verifier challenge.
pub struct OfflinePrecomputation {
    pub q_grid: Vec<Fr>,
    pub chunk_size: usize,
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

    /// PHASE 1: Pure Offline Precomputation Phase
    /// Generates the multivariate extrapolated grid `q` using solely the public polynomials.
    /// Zero interactive challenges or dynamic inputs (witnesses) are used here.
    #[inline(never)]
    pub fn precomputation_phase(
        &self,
        stream: &mut dyn PolynomialStream<Fr>,
    ) -> OfflinePrecomputation {
        let d = stream.degree();
        let early_window_size = self.early_window_size;
        let grid_size = usize::pow(d + 1, early_window_size as u32);
        let chunk_size = 1 << early_window_size;
        let total_size = 1usize << stream.num_vars();
        let num_chunks = total_size / chunk_size;

        // Lecture seule : chunk_at ne touche pas le curseur, donc partageable entre threads.
        let stream_ref: &dyn PolynomialStream<Fr> = &*stream;

        let q = (0..num_chunks)
            .into_par_iter()
            .map(|chunk_idx| {
                let chunk = stream_ref.chunk_at(chunk_idx, chunk_size);
                multi_product_eval(&chunk, d, early_window_size)
            })
            .reduce(
                || vec![Fr::ZERO; grid_size],
                |mut acc, partial| {
                    for i in 0..grid_size {
                        acc[i] += partial[i];
                    }
                    acc
                },
            );

        OfflinePrecomputation { q_grid: q, chunk_size }
    }

    /// PHASE 1 (Sequential variant): NEW ! TO UNDERSTAND
    /// Strictly identical computation to `precomputation_phase`, but executed on a single
    /// thread (no rayon `into_par_iter`/`reduce`) so it can be timed against the parallel
    /// version in `bench_offline_seq_vs_parallel`. Not meant for production use.
    #[inline(never)]
    pub fn precomputation_phase_sequential(
        &self,
        stream: &mut dyn PolynomialStream<Fr>,
    ) -> OfflinePrecomputation {
        let d = stream.degree();
        let early_window_size = self.early_window_size;
        let grid_size = usize::pow(d + 1, early_window_size as u32);
        let chunk_size = 1 << early_window_size;
        let total_size = 1usize << stream.num_vars();
        let num_chunks = total_size / chunk_size;

        let mut q = vec![Fr::ZERO; grid_size];
        for chunk_idx in 0..num_chunks {
            let chunk = stream.chunk_at(chunk_idx, chunk_size);
            let partial = multi_product_eval(&chunk, d, early_window_size);
            for i in 0..grid_size {
                q[i] += partial[i];
            }
        }

        OfflinePrecomputation { q_grid: q, chunk_size }
    }

    /// PHASE 2: Online Phase
    /// Consumes the offline grid structure, executes the interactive protocol loop 
    /// over the window, and seamlessly falls back into the linear-time execution.
    #[inline(never)]
    pub fn online_phase(
        &self,
        stream: &mut dyn PolynomialStream<Fr>,
        sumcheck_claim: Fr,
        offline: OfflinePrecomputation,
    ) -> bool {
        let num_vars = stream.num_vars();
        let d = stream.degree();
        let early_window_size = self.early_window_size;
        
        let mut verifier = Verifier::new(d);
        let mut c_i = sumcheck_claim;
        let mut rng = rand::thread_rng();
        
        let mut q = offline.q_grid;
        let chunk_size = offline.chunk_size;

        // --- 1. Emulate rounds 1 to \omega_1 using the pre-extrapolated grid ---
        let u_d_hat = get_u_hat_domain(d);
        let base = d + 1;

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
                q = dynamic_folding_step(&q, challenge, d, base, remaining_vars);
            }
        }

        // --- 2. Transition to Remaining Linear Rounds (Phase Finale) ---
        let remaining_rounds = num_vars - early_window_size;
        let final_hypercube_size = 1 << remaining_rounds;

        let mut final_prover_arrays = vec![vec![Fr::ZERO; final_hypercube_size]; d];
        stream.rewind();

        let mut element_idx = 0;
        while let Some(chunk) = stream.next_chunk(chunk_size) {
            for k in 0..d {
                let folded_scalar =
                    fold_hypercube_chunk(&chunk[k], &verifier.challenges[0..early_window_size]);
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

        // Final Space-Efficient Oracle Verification Check
        let final_evals = stream.evaluate_at_point(&verifier.challenges);
        let mut expected_product = Fr::ONE;
        for val in final_evals {
            expected_product *= val;
        }

        c_i == expected_product
    }
}

impl SumcheckProtocol<Fr> for EvalProductSV {
    fn run(&self, stream: &mut dyn PolynomialStream<Fr>, sumcheck_claim: Fr) -> bool {
        // Step 1: Purely Statically Extrapolate (Offline)
        let offline_data = self.precomputation_phase(stream);
        
        // Step 2: Handle Verification Loop and Interaction (Online)
        self.online_phase(stream, sumcheck_claim, offline_data)
    }

    // FAKE
    fn run_sb_1(&self, stream: &mut dyn PolynomialStream<Fr>, sumcheck_claim: Fr) -> bool {
        true
    }
}