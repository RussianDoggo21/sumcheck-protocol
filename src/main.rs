// First implementation of sumcheck protocol using arkworks

// Finite field F
use ark_test_curves::bls12_381::Fr;

// Polynomial poly
use ark_poly::polynomial::multivariate::{SparsePolynomial, SparseTerm, Term};
use ark_poly::DenseMVPolynomial;

// Modules import
mod utils;
mod prover;
mod verifier;
mod protocol;

use protocol::sc_protocol;
use utils::print_sc_poly_and_claim;

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

    print_sc_poly_and_claim(&poly, gamma);

    sc_protocol(&poly, gamma);
}



