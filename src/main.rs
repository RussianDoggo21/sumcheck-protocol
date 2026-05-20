// First implementation of sumcheck protocol using arkworks

// Finite field F
use ark_test_curves::bls12_381::Fr;

// Polynomial poly
use ark_poly::DenseMVPolynomial;
use ark_poly::polynomial::multivariate::{SparsePolynomial, SparseTerm, Term};

// Modules import
mod protocol;
mod prover;
mod utils;
mod verifier;

use protocol::sc_protocol;

fn main() {
    // poly(x_0, x_1, x_2) = 2*x_0 + x_0*x_2 + x_1*x_2

    let poly = SparsePolynomial::from_coefficients_vec(
        3,
        vec![
            (Fr::from(2), SparseTerm::new(vec![(0, 1)])),
            (Fr::from(1), SparseTerm::new(vec![(0, 1), (2, 1)])),
            (Fr::from(1), SparseTerm::new(vec![(1, 1), (2, 1)])),
            (Fr::from(0), SparseTerm::new(vec![])),
        ],
    );
    let gamma = Fr::from(12);
    let mut rng = rand::thread_rng();

    sc_protocol(&poly, gamma, &mut rng);
}

// To run the tests :  cargo test -- --nocapture
