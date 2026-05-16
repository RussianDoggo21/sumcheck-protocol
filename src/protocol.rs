// Contains the logic of the sumcheck protocol 

use ark_test_curves::bls12_381::Fr;
use ark_poly::polynomial::multivariate::{SparsePolynomial, SparseTerm};
use ark_poly::univariate::DensePolynomial;
use ark_poly::Polynomial;

use crate::utils::{PolyType, poly_type};
use crate::prover::prover_i;
use crate::verifier::verifier_i;



// Sumcheck protocol
pub fn sc_protocol(poly: &SparsePolynomial<Fr, SparseTerm>, gamma: Fr) -> bool {
    // 1. Test if p is multilinear (easy case) or multivariate (general case)
    let poly_type = poly_type(poly);

    // Do the actual sumcheck protocol while precising the type of poly
    let mut challenges = vec![];
    let mut current_claim = gamma;
    for round in 0..poly.num_vars {
        let check_round_i =
            sc_protocol_round(round, poly, &poly_type, &mut current_claim, &mut challenges);
        if check_round_i == false {
            return false;
        }
    }
    true
}

// i-th round of the sumcheck protocol
// poly is a multivariate polynomial which can be decomposed into a product of d multilinear polynomials
// For simplicity, we start on the case where poly is a simple multilinear polynomial ??
pub fn sc_protocol_round(
    current_round: usize,
    poly: &SparsePolynomial<Fr, SparseTerm>,
    poly_type: &PolyType,
    current_claim: &mut Fr,
    challenges: &mut Vec<Fr>,
) -> bool {
    println!("Starting round {}", current_round + 1);

    // 1) P generates the MLE g_i(X) of the current univariate polynomial p_i(X_i) = SUM_ON_ALPHAS(poly(alpha_1, ..., X_i, ... alpha_n))
    // g_i = mle(p_i)
    let g_i: DensePolynomial<Fr> = prover_i(current_round, poly, poly_type, challenges);

    // 2.1) V checks that Sum of g_i(X) over {0,1} is the current_claim (i.e g_i(0) + g_i(1) = current_claim)
    // 2.2) If that's the case, V sends a new challenge
    let w_i = match verifier_i(&g_i, *current_claim) {
        Ok(w_i) => w_i,
        Err(e) => {
            println!("{}", e);
            return false;
        }
    };
    challenges.push(w_i);

    // 3) Next round
    // 3.1) We define the next claim
    *current_claim = g_i.evaluate(&w_i);
    // 3.2) We also confirm that the verifier accepted the proof of the round
    true
}