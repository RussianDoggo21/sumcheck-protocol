// First implementation of sumcheck protocol using arkworks

// Finite field F
use ark_test_curves::bls12_381::Fr;

// Polynomial poly
use ark_poly::DenseMVPolynomial;
use ark_poly::polynomial::multivariate::{SparsePolynomial, SparseTerm, Term};

// Timer
use std::time::Instant;

// Modules import
mod utils;
mod naive;
mod improved;

use naive::protocol::sc_protocol as sc_protocol_naive;

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

    println!("Starting naive protocol");
    let start_naive = Instant::now();
    sc_protocol_naive(&poly, gamma, &mut rng);
    let duration_naive = start_naive.elapsed();
    println!("Naive protocol OK \nTime = {:?} ", duration_naive);
}

// To run the tests :  cargo test 
