// First implementation of sumcheck protocol using arkworks

// Arkworks API for sumcheck
use ark_linear_sumcheck::ml_sumcheck::MLSumcheck;

// Timer
use std::time::{Instant, Duration};

// To write the benchmark
use std::fs::File;
use std::io::Write;

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
    //test_range_monomials(200, 10);
    multilinear_test(200);
}

fn test_range_monomials(max_monomials: usize, num_runs : u32) {
    let mut file = File::create("benchmark_results.csv").expect("Unable to create file");
    
    // CORRECTION : L'en-tête est maintenant entièrement en millisecondes (ms)
    writeln!(file, "Monomials,Naive_ms,Arkworks_ms,Optimized_ms").unwrap();
    file.flush().unwrap();
    
    for num_m in (10..=max_monomials).step_by(10) {
        println!("\n==================================================");
        println!(" Benchmarking for {} monomials (Average over {} runs)...", num_m, num_runs);
        println!("==================================================");
        
        let mut total_naive = Duration::ZERO;
        let mut total_ark = Duration::ZERO;
        let mut total_opt = Duration::ZERO;

        // Boucle pour effectuer les multiples runs
        for run in 1..=num_runs {
            print!("   Run {}/{}... ", run, num_runs);
            std::io::stdout().flush().unwrap(); // Force l'affichage immédiat

            let (d_naive, d_ark, d_opt) = multilinear_test(num_m);
            
            total_naive += d_naive;
            total_ark += d_ark;
            total_opt += d_opt;
            
            println!("Done.");
        }

        // Calcul des moyennes
        let avg_naive = total_naive / num_runs;
        let avg_ark = total_ark / num_runs;
        let avg_opt = total_opt / num_runs;

        // Conversion en millisecondes (f64)
        let duration_naive_ms = avg_naive.as_secs_f64() * 1000.0;
        let duration_ark_ms = avg_ark.as_secs_f64() * 1000.0;
        let duration_opt_ms = avg_opt.as_secs_f64() * 1000.0;

        // Écriture propre dans le CSV
        writeln!(file, "{},{},{},{}", num_m, duration_naive_ms, duration_ark_ms, duration_opt_ms).unwrap();
        file.flush().unwrap();
        
        println!("\n -> Average Naive    : {:.2} ms", duration_naive_ms);
        println!(" -> Average Arkworks : {:.2} ms", duration_ark_ms);
        println!(" -> Average Optimized: {:.4} ms", duration_opt_ms);
    }
    println!("\n[OK] Fichier CSV généré avec succès sous le nom 'benchmark_results.csv' !");
}

fn multilinear_test(num_m : usize) -> (Duration, Duration, Duration){

    let mut rng = rand::thread_rng();
    let (poly0, list_of_products) = generate_poly_test(&mut rng, num_m); 
    //println!("{}\n", format_multivariate_sparse_poly(&poly0));
/* ************************************************************************************************************************************************************** */   
    
    println!("\nStarting naive protocol");
    let gamma = compute_hypercube_sum(&poly0);
    let start_naive = Instant::now();
    sc_protocol_naive(&poly0, gamma, &mut rng);
    let duration_naive = start_naive.elapsed();
    println!("\nNaive protocol OK \nTime = {:?} ", duration_naive);

/*************************************************************************************************************************************************************** */

    println!("\nStarting arkworks protocol");
    let start_arkworks = Instant::now();
    let proof = MLSumcheck::prove(&list_of_products)
        .expect("The arkworks prover failed");
    let duration_arkworks = start_arkworks.elapsed();
    let claimed_sum = MLSumcheck::extract_sum(&proof);
    println!("Arkworks protocol OK \nTime = {:?}", duration_arkworks);

/* ************************************************************************************************************************************************************** */

    println!("\nStarting optimized (Small-Value) protocol");
    let evals_small = generate_small_evaluations_from_poly(&poly0);
    let start_improved = Instant::now();
    let (sum_improved, _proofs) = sc_protocol_improved(poly0.num_vars, &evals_small);
    let duration_improved = start_improved.elapsed(); 
    println!("Optimized protocol OK \nTime = {:?}", duration_improved);

/* ************************************************************************************************************************************************************** */
    assert!(
        gamma == claimed_sum && claimed_sum == sum_improved,
        "Error : Computed sums not equal... Naive: {}, Arkworks: {}, Improved: {}",
        gamma, claimed_sum, sum_improved
    );

    (duration_naive, duration_arkworks, duration_improved)
}
// To run more tests on the naive protocol :  cargo test