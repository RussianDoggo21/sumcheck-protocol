// Contains the logic of the verifier
use ark_poly::Polynomial;
use ark_poly::univariate::DensePolynomial;
use ark_std::rand::Rng;
use ark_test_curves::bls12_381::Fr;
use ark_poly::polynomial::multivariate::{SparsePolynomial, SparseTerm};

// Verifier algorithm at the i-th round (from i=0 to i=n-1)
// p_i(0) + p_i(1) =? current_claim 
pub fn verifier_i<R: Rng>(
    p_i: &DensePolynomial<Fr>,
    current_claim: Fr,
    rng: &mut R,
) -> Result<Fr, &'static str> {
        let eval_0 = p_i.evaluate(&Fr::from(0));
        let eval_1 = p_i.evaluate(&Fr::from(1));
        let check_sum = eval_0 + eval_1;

        // If the check fails, we stop the programm : the Prover is cheating
        if check_sum != current_claim {
            return Err("Sumcheck verification failed: g_i(0) + g_i(1) != current_claim");
        };

        // If the check passes, V "sends" a random field element w_i

        // TEMPORARY CODE
        let random_small_int: u64 = rng.gen_range(0..=100); // TO DELETE
        let w_i = Fr::from(random_small_int); // TO REPLACE WITH Fr::rand(&mut rng)
        // TEMPORARY CODE

        Ok(w_i)

}

// Final check (i=n)
// poly(w_0,...,w_n-1)=?final_claim
// with final_claim = p_n(w_n-1) (the last element of challenges)
pub fn verifier_final(poly : &SparsePolynomial<Fr, SparseTerm>, final_claim : Fr,challenges: &mut Vec<Fr>) -> bool {
    let final_evaluation = poly.evaluate(&challenges);
    if final_evaluation != final_claim {
        println!("Protocol Failed: Final evaluation mismatch!");
        return false;
    }
    true
}
