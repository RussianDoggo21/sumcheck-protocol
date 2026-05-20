// Contains the logic of the prover
use ark_poly::DenseUVPolynomial;
use ark_poly::polynomial::multivariate::{SparsePolynomial, SparseTerm};
use ark_poly::univariate::DensePolynomial;
use ark_test_curves::bls12_381::Fr;

use crate::utils::{PolyType, p_i_coeff};

// Prover algorithm at the i-th round
pub fn prover_i(
    current_round: usize,
    poly: &SparsePolynomial<Fr, SparseTerm>,
    poly_type: &PolyType,
    challenges: &mut Vec<Fr>,
) -> DensePolynomial<Fr> {
    let result = match poly_type {
        PolyType::Multilinear => prover_i_multilinear(current_round, poly, challenges),
        PolyType::Multivariate(_d) => prover_i_multivariate(current_round, poly, challenges),
    };
    result
}

pub fn prover_i_multilinear(
    current_round: usize,
    poly: &SparsePolynomial<Fr, SparseTerm>,
    challenges: &mut Vec<Fr>,
) -> DensePolynomial<Fr> {
    // 1) P computes the dense univariate polynomial p_i(X) defined on Fr[X]

    // 1.1) Computation of the coefficients a and b such that p_i(X) = a.X + b
    let (a, b) = p_i_coeff(current_round, poly, challenges);
    // 1.2) Generation of p_i(X)
    let p_i = DensePolynomial::from_coefficients_vec(vec![b, a]);
    p_i
}

/// A METTRE A JOUR
pub fn prover_i_multivariate(
    _current_round: usize,
    _poly: &SparsePolynomial<Fr, SparseTerm>,
    _challenges: &mut Vec<Fr>,
) -> DensePolynomial<Fr> {
    todo!("Le cas général multivarié n'est pas encore implémenté !");
}
