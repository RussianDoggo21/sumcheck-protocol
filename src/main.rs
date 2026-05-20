// First implementation of sumcheck protocol using arkworks

// Arkworks API for sumcheck
use ark_linear_sumcheck::ml_sumcheck::MLSumcheck;

// Timer
use std::time::Instant;

// Modules import
mod utils;
mod naive;
mod improved;

use naive::protocol::sc_protocol as sc_protocol_naive;

use crate::utils::{compute_hypercube_sum, format_multivariate_sparse_poly, generate_poly_test};

fn main() {
    test(3);
}

fn test(number_of_tests : usize){

    for i in 0..number_of_tests {
        println!("\n==================================================== Test {} ====================================================", i+1);
        let mut rng = rand::thread_rng();
        let (poly0, list_of_products) = generate_poly_test(&mut rng); 
        println!("{}\n", format_multivariate_sparse_poly(&poly0));
    /* ************************************************************************************************************************************************************** */   
        
        println!("Starting naive protocol");
        let gamma = compute_hypercube_sum(&poly0);
        let start_naive = Instant::now();
        sc_protocol_naive(&poly0, gamma, &mut rng);
        let duration_naive = start_naive.elapsed();
        println!("Naive protocol OK \nTime = {:?} ", duration_naive);
        println!("Computed sum = {}\n", gamma);

    /*************************************************************************************************************************************************************** */

        println!("Starting arkworks protocol");
        let start_arkworks = Instant::now();
        let proof = MLSumcheck::prove(&list_of_products)
            .expect("The arkworks prover failed");
        let duration_arkworks = start_arkworks.elapsed();
        let claimed_sum = MLSumcheck::extract_sum(&proof);
        println!("Arkworks protocol OK \nTime = {:?}", duration_arkworks);
        println!("Computed sum = {}",  ark_ff::PrimeField::into_bigint(claimed_sum));
    /* ************************************************************************************************************************************************************** */

    }
}
// To run more tests on the naive protocol :  cargo test 
