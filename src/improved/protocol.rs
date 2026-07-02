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
}

impl SumcheckProtocol<Fr> for EvalProductSV {
    fn run(&self, stream: &mut dyn PolynomialStream<Fr>, sumcheck_claim: Fr) -> bool {
        let num_vars = stream.num_vars();
        let d = stream.degree();
        let early_window_size = self.early_window_size; // Denoted omega_1 in the comments

        // Persistent unique Verifier instance throughout all protocol phases
        let mut verifier = Verifier::new(d);
        let mut c_i = sumcheck_claim;
        let mut rng = rand::thread_rng();

        // -------------------------------------------------------------------------
        // STEP 1.(a) : Compute the intermediate grid polynomial q
        // -------------------------------------------------------------------------
        // q lives on the grid U_d^{\omega_1}, so its size is (d + 1)^\omega_1
        let grid_size = usize::pow(d + 1, early_window_size as u32);
        let mut q = vec![Fr::ZERO; grid_size];

        let chunk_size = 1 << early_window_size; // Size of each sub-cube chunk: 2^\omega_1
        stream.rewind();

        // On each loop, we fix a x_prime in {0, 1}^{l - omega_1} (size 2^l-omega_1)
        // In practice, it comes back to varying the 2^omega_1 other values of each p_k (this is the variable chunk)
        // Reminder : each p_k is represend by its 2^l evaluations over {0,1}^l

        // We then compute p_x_prime = product(k=1 to d)[poly_k(X_1, ..., X_omega_1, x_prime)]
        // We need ALL the p_x_prime (in practice, all the chunks) to compute q
        // We need to obtain #[{0,1}^(l-omega_1)] = 2^(l-omega_1) chunks of size #[{0,1}^omega_1] = 2^omega_1
        // in order to obtain the 2^l evaluations we need to compute q in its whole
        while let Some(chunk) = stream.next_chunk(chunk_size) {
            // Compute the multi-product evaluation p_x_prime for the current chunk of size 2^omega_1
            // p_x_prime is a grid of size (d + 1)^\omega_1 (evaluations over U_d^omega_1)
            let p_x_prime = multi_product_eval(&chunk, d, early_window_size);

            // Accumulate (sum over x_prime) into our q polynomial grid (U_d^omega_1)
            for i in 0..grid_size {
                q[i] += p_x_prime[i];
            }
        }

        // -------------------------------------------------------------------------
        // STEP 1.(b) : Emulate rounds 1 to omega_1 on the grid q
        // Sum(x'' in {0,1}^w_1)[q(x'')] = Sum(x in {0,1}^l)[g(x)] = sumcheck_claim
        // -------------------------------------------------------------------------

        let u_d_hat = get_u_hat_domain(d);
        // "Emulate rounds [S_t + 1; S_t + w_t]"
        // Here, S_t = 0 and w_t = early_window_size (t=1)
        for i in 0..early_window_size {
            // The restricted domain U_d_hat contains d elements: { \infty, 1, ..., d-1 }
            // Goal : to compute s_i(X) by sending the d evaluations s_i(u), with u in U_d_hat
            let mut s_i_evals = vec![Fr::ZERO; d];
            let remaining_vars = early_window_size - 1 - i;
            let base = d + 1;

            // Number of boolean combinations for the future variables (j > i)
            let next_hypercube_size = 1 << remaining_vars;

            // Extract s_i(u) for each point u in U_d_hat using clean domain abstraction
            for (u_idx, &u) in u_d_hat.iter().enumerate() {
                let mut sum_over_hypercube = Fr::ZERO;

                // Map the EvaluationPoint directly to its physical position inside the grid q
                let current_var_idx = match u {
                    EvaluationPoint::Infinity => 0,
                    EvaluationPoint::Value(v) => (v + 1) as usize,
                };

                for x_prime in 0..next_hypercube_size {
                    // Compute the memory offset for the future boolean variables (j > i)
                    // We map each bit of x_prime to index 1 (value 0) or index 2 (value 1)
                    let mut future_offset = 0;
                    for var_j in 0..remaining_vars {
                        let bit = (x_prime >> var_j) & 1;
                        let grid_val_idx = if bit == 0 { 1 } else { 2 };
                        future_offset += grid_val_idx * usize::pow(base, (var_j + 1) as u32);
                    }

                    // The active variable is always at stride 1 thanks to dynamic folding
                    let final_grid_index = current_var_idx + future_offset;
                    sum_over_hypercube += q[final_grid_index];
                }

                s_i_evals[u_idx] = sum_over_hypercube;
            }

            // Send s_i evaluations to the Verifier
            verifier.add_s_i(s_i_evals);

            // Verifier updates and challenge generation
            let challenge = verifier.send_challenge(&mut rng);
            let s_i_0 = verifier.compute_s_i_0(c_i);
            c_i = verifier.update_c_i(challenge, s_i_0);

            // -------------------------------------------------------------------------
            // OPTIMIZED DYNAMIC FOLDING (Infinity-Aware): Profiled via engine::dynamic_folding_step
            // -------------------------------------------------------------------------
            if remaining_vars > 0 {
                q = dynamic_folding_step(&q, challenge, d, base, remaining_vars);
            }
        }

        // -------------------------------------------------------------------------
        // PHASE 3 (STEP 2): Transition to the Remaining Linear Rounds
        // -------------------------------------------------------------------------

        // Size of the hypercube for the remaining variables (from \omega_1 + 1 to l)
        let remaining_rounds = num_vars - early_window_size;
        let final_hypercube_size = 1 << remaining_rounds;

        // Compact tables to store the folded evaluations of each of the d polynomials
        let mut final_prover_arrays = vec![vec![Fr::ZERO; final_hypercube_size]; d];

        // Rewind the stream to read from the beginning
        stream.rewind();

        let mut element_idx = 0;
        // The chunk size matches the size of the early window hypercube (2^\omega_1)
        while let Some(chunk) = stream.next_chunk(chunk_size) {
            for k in 0..d {
                // Collapse the current chunk of size 2^\omega_1 down to a single scalar
                // by sequentially applying the collected challenges from the first window
                let folded_scalar =
                    fold_hypercube_chunk(&chunk[k], &verifier.challenges[0..early_window_size]);

                // Place the collapsed scalar into the compact bookkeeping array
                final_prover_arrays[k][element_idx] = folded_scalar;
            }
            element_idx += 1;
        }

        // Bootstrap the linear-time Prover using the memory-compact compressed tables
        let mut prover = Prover::with_arrays(final_prover_arrays);

        // Standard Sumcheck execution loop for the remaining rounds (\omega_1 to l)
        for i in early_window_size..num_vars {
            // Compute s_i(u) over U_hat using the compressed bookkeeping arrays
            let s_i = prover.compute_s_i(num_vars, i);
            verifier.add_s_i(s_i);

            // Fetch challenge and update the sumcheck claim target C_i
            let s_i_0 = verifier.compute_s_i_0(c_i);
            let challenge = verifier.send_challenge(&mut rng);

            c_i = verifier.update_c_i(challenge, s_i_0);

            // Linear-time in-place update of the compressed arrays for the next round
            prover.update_p_arrays(num_vars, i, challenge);
        }

        // -------------------------------------------------------------------------
        // PHASE 4: Final Space-Efficient Oracle Verification Check
        // -------------------------------------------------------------------------
        // Re-read the stream point evaluations using the full random challenge vector
        let final_evals = stream.evaluate_at_point(&verifier.challenges);

        // Compute the final expected product: \prod_{k=1}^d p_k(r_1, ..., r_l)
        let mut expected_product = Fr::ONE;
        for val in final_evals {
            expected_product *= val;
        }

        // The verifier accepts if and only if the final claim matches the oracle product evaluation
        c_i == expected_product
    }
}
