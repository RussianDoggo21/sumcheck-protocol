// First implementation of sumcheck protocol using arkworks

// Arkworks API for sumcheck
use ark_linear_sumcheck::ml_sumcheck::MLSumcheck;

// Timer
use std::time::Instant;

// Modules import
mod utils;
mod naive;
mod improved {
    pub mod arithmetic;
    pub mod protocol;
}

use naive::protocol::sc_protocol as sc_protocol_naive;
use improved::protocol::sc_protocol_improved;
use crate::utils::{compute_hypercube_sum, format_multivariate_sparse_poly, generate_poly_test, generate_small_evaluations_from_poly};

fn main() {
    multilinear_test(3);
}

fn multilinear_test(number_of_tests : usize){

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
        println!("Computed sum = {}\n",  ark_ff::PrimeField::into_bigint(claimed_sum));

    /* ************************************************************************************************************************************************************** */

        println!("Starting optimized (Small-Value) protocol");
        let evals_small = generate_small_evaluations_from_poly(&poly0);
        let start_improved = Instant::now();
        let (sum_improved, _proofs) = sc_protocol_improved(poly0.num_vars, &evals_small);
        let duration_improved = start_improved.elapsed(); 
        println!("Optimized protocol OK \nTime = {:?}", duration_improved);
        println!("Computed sum = {}", sum_improved);

    /* ************************************************************************************************************************************************************** */

    }
}
// To run more tests on the naive protocol :  cargo test