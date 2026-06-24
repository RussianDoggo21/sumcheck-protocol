use ark_poly::DenseMultilinearExtension;
use ark_test_curves::bls12_381::Fr;
use ark_ff::PrimeField;
use crate::improved::prover::Prover;
use crate::improved::verifier::Verifier;
use crate::improved::streaming::PolynomialStream;

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