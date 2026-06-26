use ark_poly::DenseMultilinearExtension;
use ark_test_curves::bls12_381::Fr;
use ark_ff::{Field, PrimeField};
use crate::improved::prover::Prover;
use crate::improved::verifier::Verifier;
use crate::improved::streaming::PolynomialStream;
use crate::improved::engine::multi_product_eval;

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
        let full_chunk = stream.next_chunk(1 << num_vars)
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

        assert!(list_of_poly.len() > 0, "Cannot run sumcheck on an empty list of polynomials");

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

        Self { early_window_size: omega_1 }
    }

}

impl SumcheckProtocol<Fr> for EvalProductSV {
    fn run(&self, stream: &mut dyn PolynomialStream<Fr>, sumcheck_claim: Fr) -> bool {
        let num_vars = stream.num_vars();
        let d = stream.degree();
        let early_window_size = self.early_window_size; // Denoted omega_1 in the comments
        
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
        // -------------------------------------------------------------------------
        // Here, we need to run a "grid-based linear-time emulation" (Appendix C)
        // to consume the first \omega_1 rounds and collect the first challenges r_1, ..., r_{\omega_1}.
        
        // TODO: Implémenter l'émulation sur grille
        
        // -------------------------------------------------------------------------
        // STEP 2 : Final phase for remaining rounds
        // -------------------------------------------------------------------------
        // For the remaining rounds (from omega_1 + 1 to l), the paper states we can
        // stream and store the folded polynomials, then run standard LinearTime_SC.
        
        // TODO: Gérer la phase finale linéaire
        
        true
    }
}