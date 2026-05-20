// Contains the logic of the sumcheck protocol

use ark_poly::Polynomial;
use ark_poly::polynomial::multivariate::{SparsePolynomial, SparseTerm};
use ark_poly::univariate::DensePolynomial;
use ark_std::rand::Rng;
use ark_test_curves::bls12_381::Fr;

use crate::naive::prover::prover_i;
use crate::utils::{PolyType, poly_type, format_univariate_dense_poly, format_multivariate_sparse_poly};//, print_round_status, print_final_round_status};
use crate::naive::verifier::{verifier_final, verifier_i};

// For debugging
use tracing::debug;

// Sumcheck protocol
pub fn sc_protocol<R: Rng>(
    poly: &SparsePolynomial<Fr, SparseTerm>,
    gamma: Fr,
    rng: &mut R,
) -> bool {

    debug!("=== {:} ===", format_multivariate_sparse_poly(poly));
    debug!("=== Original claim (gamma) = {:?}", gamma);

    // 1. Test if p is multilinear (easy case) or multivariate (general case)
    let poly_type = poly_type(poly);

    // 2. Do the actual sumcheck protocol while precising the type of poly
    let mut challenges = vec![];
    let mut current_claim = gamma;

    for round in 0..poly.num_vars+1 {
        let check_round_i = sc_protocol_round(
            round,
            poly,
            &poly_type,
            &mut current_claim,
            &mut challenges,
            rng,
        );
        if check_round_i == false {
            println!("Check failure on round {}", round+1);
            return false;
        }
    }

    true
}

// i-th round of the sumcheck protocol
// poly is a multivariate polynomial which can be decomposed into a product of d multilinear polynomials
// For simplicity, we start on the case where poly is a simple multilinear polynomial ??
pub fn sc_protocol_round<R: Rng>(
    current_round: usize,
    poly: &SparsePolynomial<Fr, SparseTerm>,
    poly_type: &PolyType,
    current_claim: &mut Fr,
    challenges: &mut Vec<Fr>,
    rng: &mut R,
) -> bool {
    if current_round<poly.num_vars{
        // 1) P generates univariate polynomial p_i(X_i) = SUM_ON_ALPHAS(poly(alpha_1, ..., X_i, ... alpha_n))
        let p_i: DensePolynomial<Fr> = prover_i(current_round, poly, poly_type, challenges);


        // 2) V checks that Sum of g_i(X) over {0,1} is the current_claim (i.e g_i(0) + g_i(1) = current_claim)
        let w_i = match verifier_i(&p_i, *current_claim, rng) {
        
            Ok(w_i) => w_i,  // 2.1) If that's the case, V sends a new challenge
            
            // 2.2) Else, we return false : the sumcheck has failed
            Err(e) => {
                tracing::error!("{}", e);
                return false;
            }
        };    

        // 3) Preparation for next round
        
        challenges.push(w_i); // 3.1) We update the list of challenges

        // 3.2) We define the claim for the next round
        let old_claim = *current_claim;
        let next_claim = p_i.evaluate(&w_i);
        *current_claim = next_claim;

        // ============== LOGS FOR THE ROUND'S DEBUG  ==============
        debug!("=== DEBUG ROUND {} ===", current_round);
        debug!("  Current Claim V expects: {:?}", old_claim);
        debug!("  Current challenges: {:?}", &challenges[..challenges.len()-1]);
        debug!("  {:?}", format_univariate_dense_poly(&p_i, current_round));
        debug!("  p_{}(0) = {:?}", current_round, p_i.evaluate(&Fr::from(0)));
        debug!("  p_{}(1) = {:?}", current_round, p_i.evaluate(&Fr::from(1)));
        debug!("  p_{}(0) + p_{}(1) = {:?}", current_round, current_round, p_i.evaluate(&Fr::from(0))+p_i.evaluate(&Fr::from(1)));
        debug!("  New challenge = {:?}", w_i);
        debug!("  Next claim: p_{}({:?}) = {:?}", current_round, w_i, next_claim);
        // ========================================================
    
        true // 3.4) We also confirm that the verifier accepted the proof of the round
    }    
    else {
        let final_result = verifier_final(poly, *current_claim, challenges);
        // =========== LOGS FOR THE FINAL CHECK ================
        debug!("=== DEBUG OF FINAL ROUND ===");
        debug!("  Final claim to check by V: p_{}(w_{}) = p_{}({:?}) = {:?}",poly.num_vars, poly.num_vars, poly.num_vars, &challenges[challenges.len()-1],current_claim);
        debug!("  Final evaluation: poly(w_0, ...,w_{}) = {:?}", poly.num_vars, poly.evaluate(challenges));
        // =====================================================
        final_result
    }

}

#[cfg(test)] // This module is only compiled when running 'cargo test'
mod tests {
    use itertools::Itertools;

    use ark_poly::multivariate::Term;
    use ark_poly::polynomial::multivariate::{SparsePolynomial, SparseTerm};
    use ark_poly::{DenseMVPolynomial, Polynomial};
    use ark_std::rand::Rng;
    use ark_test_curves::bls12_381::Fr;

    use super::*;
    use crate::utils::{i_to_boolean_point};

    //use tracing::{info, debug};

    /// Helper function to automatically compute the sum of a polynomial over the Boolean hypercube {0,1}^n
    fn compute_hypercube_sum(poly: &SparsePolynomial<Fr, SparseTerm>) -> Fr {
        let num_points = 1 << poly.num_vars; // 2^n points
        let mut sum = Fr::from(0);

        for i in 0..num_points {
            // Generate the i-th point of the hypercube (e.g., [0, 1, 0])
            let point = i_to_boolean_point(i, poly.num_vars);
            // Evaluate the polynomial at this specific point and add it to the total
            sum += poly.evaluate(&point);
        }
        sum
    }

    #[test]
    fn test_sumcheck_multilinear() {
        // INITIALISATION OF THE LISTENER OF LOGS
        let _ = tracing_subscriber::fmt()
        .with_env_filter("debug") // On écoute tout ce qui est niveau debug ou supérieur
        .with_target(false)       // Optionnel : masque le nom du fichier pour un log plus propre
        .try_init();

        println!("\n--- Launching Sumcheck Protocol Test (multilinear testing) ---");

        // 1. Randomly generates :
        // - the number of variables
        // - the number of monomials
        // - each monomial term and coefficient

        // Random number generator
        let mut rng = rand::thread_rng();

        // Number of variables
        let n: usize = rng.gen_range(2..=10);
        println!("--- Generating a multilinear polynomial with {n} variables ---");

        // Generation of all monomial possibles
        let mut all_monomials: Vec<Vec<usize>> = (0..n).powerset().collect();

        // Number of monomials
        let num_monomial = rng.gen_range(2..=8);
        let mut terms = Vec::with_capacity(num_monomial);

        // Generating each monomial (coeff, terms) on the fly
        for _ in 1..num_monomial {
            // Coefficient
            let coeff = Fr::from(rng.gen_range(1..=10));

            // Terms
            let term = if let Some(var_index_vec) = all_monomials.pop() {
                // Generation of the vector (coeff, term)
                // with term being itself a vector of (index, power)
                // In our case power is always equal to 1
                let term_vec: Vec<(usize, usize)> = var_index_vec
                    .into_iter()
                    .map(|index_var| (index_var, 1)) // Transform each index into (index, 1)
                    .collect(); // Gather everything into a Vec

                // 3. Create the SparseTerm
                SparseTerm::new(term_vec)
            } else {
                continue;
            };
            terms.push((coeff, term));
        }

        let poly = SparsePolynomial::from_coefficients_vec(n, terms);

        // 2. Automatically compute gamma over the hypercube {0,1}^n
        let gamma = compute_hypercube_sum(&poly);

        // 3. Run the sumcheck protocol
        let result = sc_protocol(&poly, gamma, &mut rng);

        // 4. Assert success
        assert!(
            result,
            "The sumcheck protocol failed but it was expected to succeed!"
        );
    }
}
