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

pub trait SumcheckProtocol<F: PrimeField> {
    /// Runs the complete Sumcheck protocol (Prover + Verifier interaction).
    /// Returns `true` if the verifier accepts the proof, `false` otherwise.
    ///
    /// # Arguments
    /// * `stream` - The polynomial evaluation stream (can be MockStream or a real file stream)
    /// * `sumcheck_claim` - The initial claimed sum C_0
    fn run(&self, stream: &mut dyn PolynomialStream<F>, sumcheck_claim: F) -> bool;
}

/// The baseline Linear-Time Sumcheck protocol (RAM-heavy version).
/// It implements `SumcheckProtocol` by loading the entire stream into memory
/// to maintain compatibility with legacy structures.
pub struct LinearTimeSC;

impl SumcheckProtocol<Fr> for LinearTimeSC {
    fn run(&self, stream: &mut dyn PolynomialStream<Fr>, sumcheck_claim: Fr) -> bool {
        let num_vars = stream.num_vars();
        let num_poly = stream.degree();
        stream.rewind();

        // 1. Logistics: Extract the full Boolean hypercube (size 2^l) from the stream
        // `full_chunk` is a Vec<Vec<Fr>> containing `num_poly` sub-vectors of size 2^num_vars
        let full_chunk = stream
            .next_chunk(1 << num_vars)
            .expect("Stream should provide the full hypercube data");

        // 2. Reconstruction: Rebuild the DenseMultilinearExtension array expected by the original code
        let list_of_poly: Vec<DenseMultilinearExtension<Fr>> = full_chunk
            .into_iter()
            .map(|evals| DenseMultilinearExtension::from_evaluations_vec(num_vars, evals))
            .collect();

        // 3. Delegation: Forward arguments seamlessly to the original verified implementation
        Self::linear_time_sc(&list_of_poly, num_poly, sumcheck_claim)
    }
}

impl LinearTimeSC {
    /// The original standalone Linear-Time Sumcheck function.
    /// It has been encapsulated here as an associated method with zero internal modifications.
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

        // Initialisation of both prover and verifier via their respective constructor
        let mut prover = Prover::new(list_of_poly);
        let mut verifier = Verifier::new(num_poly);

        // C_0 is our initial sumcheck claim
        let mut c_i = sumcheck_claim;

        // Unique initialization of the rng to generate random challenges
        let mut rng = rand::thread_rng();

        // Loop through each round i = 0 ... num_vars - 1 (Rounds 1 to l)
        for i in 0..num_vars {
            // 1. Prover computes the s_i evaluations and sends them to Verifier
            let s_i = prover.compute_s_i(num_vars, i);
            verifier.add_s_i(s_i);

            // 2. Verifier computes s_i(0) and samples a random challenge r_i
            let s_i_0 = verifier.compute_s_i_0(c_i);
            let challenge = verifier.send_challenge(&mut rng);

            // 3. Verifier updates their local target claim: C_i = s_i(r_i)
            c_i = verifier.update_c_i(challenge, s_i_0);

            // 4. Prover folds their internal bookkeeping tables down by half using r_i
            prover.update_p_arrays(num_vars, i, challenge);
        }

        // Final oracle evaluation check at the end of the protocol
        verifier.final_check(list_of_poly, c_i)
    }
}

/// The Small-Value (SV) optimized Sumcheck protocol (Figure 2).
/// It features a single early window of size omega_1 to minimize runtime,
/// then falls back to linear-time sumcheck execution for remaining rounds.
pub struct EvalProductSV {
    pub early_window_size: usize,
}
/// Stores the intermediate state produced during the precomputation phase,
/// which is required to bootstrap the final linear-time phase.
pub struct PrecomputationOutput {
    pub verifier: Verifier,
    pub c_i: Fr,
    pub chunk_size: usize,
}

impl EvalProductSV {
    /// Creates a new EvalProductSV instance with an optimal early window size v*
    /// Automatically computes the optimal early window size v* based on Lemma 5.
    ///
    /// # Arguments
    /// * `d` - The number of polynomials to multiply (degree)
    /// * `num_vars` - The total number of variables (l) in the protocol
    pub fn new(d: usize, num_vars: usize) -> Self {
        // For BLS12-381, the prime field uses 6 limbs of 64-bits (381 bits total).
        // According to footnote 11: \kappa \approx 2N^2 + N. For N = 6, \kappa \approx 78.

        // TO GENERALIZE FOR ALL FIELDS
        let kappa = 78.0;

        let d_f = d as f64;

        // Calculate the argument inside the logarithm: d^2 * \kappa
        let argument = d_f * d_f * kappa;

        // Change of base formula for log_{d+1}(x) = ln(x) / ln(d + 1)
        let base = d_f + 1.0;
        let v_star_exact = argument.ln() / base.ln();

        // Round to the nearest integer as suggested by "clipped/rounded to valid rounds"
        let mut omega_1 = v_star_exact.round() as usize;

        // Clip the window size to make sure it is at least 1 and at most `num_vars`
        if omega_1 < 1 {
            omega_1 = 1;
        }
        if omega_1 > num_vars {
            omega_1 = num_vars;
        }

        Self {
            early_window_size: omega_1,
        }
    }
    /// PHASE 1 & 2: Precomputation Phase
    /// Builds the intermediate grid polynomial q and emulates the first \omega_1 rounds.
    #[inline(never)]
    pub fn precomputation_phase(
        &self,
        stream: &mut dyn PolynomialStream<Fr>,
        sumcheck_claim: Fr,
    ) -> PrecomputationOutput {
        let d = stream.degree();
        let early_window_size = self.early_window_size;

        let mut verifier = Verifier::new(d);
        let mut c_i = sumcheck_claim;
        let mut rng = rand::thread_rng();

        // 1.(a) Compute the intermediate grid polynomial q
        let grid_size = usize::pow(d + 1, early_window_size as u32);
        let mut q = vec![Fr::ZERO; grid_size];
        let chunk_size = 1 << early_window_size;
        stream.rewind();

        while let Some(chunk) = stream.next_chunk(chunk_size) {
            let p_x_prime = multi_product_eval(&chunk, d, early_window_size);
            for i in 0..grid_size {
                q[i] += p_x_prime[i];
            }
        }

        // 1.(b) Emulate rounds 1 to \omega_1 on the grid q
        let u_d_hat = get_u_hat_domain(d);
        for i in 0..early_window_size {
            let mut s_i_evals = vec![Fr::ZERO; d];
            let remaining_vars = early_window_size - 1 - i;
            let base = d + 1;
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

        PrecomputationOutput {
            verifier,
            c_i,
            chunk_size,
        }
    }

    /// PHASE 3 & 4: Final Phase
    /// Collapses the streaming data using accumulated challenges and executes the remaining linear rounds.
    #[inline(never)]
    pub fn final_phase(
        &self,
        stream: &mut dyn PolynomialStream<Fr>,
        precomp: PrecomputationOutput,
    ) -> bool {
        let num_vars = stream.num_vars();
        let d = stream.degree();
        let early_window_size = self.early_window_size;
        
        let mut verifier = precomp.verifier;
        let mut c_i = precomp.c_i;
        let chunk_size = precomp.chunk_size;
        let mut rng = rand::thread_rng();

        // Transition to Remaining Linear Rounds
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
        // En isolant les étapes ainsi, run() orchestre simplement le flux
        let precomp = self.precomputation_phase(stream, sumcheck_claim);
        self.final_phase(stream, precomp)
    }
}