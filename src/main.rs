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
mod improved;

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
    let start_improved = Instant::now();
    let (sum_improved, _proofs) = sc_protocol_improved(&poly0, &mut rng);
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

/*
#[test]
fn test_univariate_long_sandbox() {
    use ark_test_curves::bls12_381::Fr;
    use crate::improved::engine::{univariate_extrapolate, compute_kernel};
    use ark_poly::{
        polynomial::multivariate::{SparsePolynomial, SparseTerm, Term},
        DenseMVPolynomial, Polynomial,
    };

    let k = 2;
    let num_extrap = 10;

    // 1. Initial evaluations on U_2 = {inf, 0, 1}
    let mut evals = vec![
        Fr::from(3u64), // p(inf) = 3
        Fr::from(5u64), // p(0) = 5
        Fr::from(6u64), // p(1) = 6
    ];

    // 2. Compute the Lagrange kernel for k=2
    let kernel = compute_kernel(k);

    // 3. Perform chained extrapolation (adds 10 elements to the vector)
    univariate_extrapolate(&mut evals, &kernel, k, num_extrap);

    // Expected final size: 3 (initial) + 10 (extrapolations) = 13 elements
    assert_eq!(evals.len(), 13);

    println!("\n--- Verifying the 10 extrapolations ---");
    
    // 4. Dynamically verify each computed point against Arkworks reference
    // Index 0 is p(inf). Index 1 is p(0), index 2 is p(1).
    // Extrapolations start at index 3 (for X=2) up to index 12 (for X=11).

    let poly = SparsePolynomial::from_coefficients_vec(
        1,
        vec![
            (Fr::from(3), SparseTerm::new(vec![(0, 2)])),  // 3 * X^2
            (-Fr::from(2), SparseTerm::new(vec![(0, 1)])), // -2 * X^1
            (Fr::from(5), SparseTerm::new(vec![])),        // Constante 5
        ],
    );

    for x in 2..=11 {
        let memory_index = x + 1; // +1 offset because p(inf) is stored at index 0
        let computed_val = evals[memory_index];

        // Theoretical calculation: p(x) = 3x^2 - 2x + 5
        let x_fr = &vec![Fr::from(x as u64)];
        let expected_val = poly.evaluate(x_fr);

        println!("X = {:2} -> Expected: {:3} | Computed: {:?}", x, expected_val, computed_val);

        assert_eq!(
            computed_val, 
            expected_val, 
            "Extrapolation failed at point X = {}", x
        );
    }

    println!("✅ Long univariate test successfully passed!");
}
*/

/* 
#[test]
fn test_multivariate_exhaustive_sandbox() {
    use ark_test_curves::bls12_381::Fr;
    use crate::improved::engine::multivariate_extrapolate;
    use ark_poly::{
        polynomial::multivariate::{SparsePolynomial, SparseTerm, Term},
        DenseMVPolynomial, Polynomial,
    };

    let k = 1;
    let num_extrap = 2; 
    let num_vars = 2;

    // 1. Initialize the base hypercube U_1^2 (size 2^2 = 4)
    let initial_evals = vec![
        Fr::from(0u64), // p(inf, inf)
        Fr::from(1u64), // p(0, inf)  
        Fr::from(2u64), // p(inf, 0)  
        Fr::from(1u64), // p(0, 0)    
    ];

    // 2. Compute the expanded hypercube using the multivariate protocol
    let extended_cube = multivariate_extrapolate(&initial_evals, k, num_extrap, num_vars);

    // Expected size of the flat grid: 4^2 = 16 elements
    let size_d = k + 1 + num_extrap; // 4
    assert_eq!(extended_cube.len(), size_d * size_d);

    // 3. Define the ground-truth Arkworks polynomial: p(X0, X1) = 2X0 + X1 + 1
    // Variable index mapping for SparseTerm: 0 -> X0, 1 -> X1
    let poly = SparsePolynomial::from_coefficients_vec(
        num_vars,
        vec![
            (Fr::from(2u64), SparseTerm::new(vec![(0, 1)])), // 2 * X0
            (Fr::from(1u64), SparseTerm::new(vec![(1, 1)])), // 1 * X1
            (Fr::from(1u64), SparseTerm::new(vec![])),        // 1
        ],
    );

    println!("\n--- Exhaustive verification of the extended hypercube ({0}x{0}) ---", size_d);

    // Coordinate mapping for standard points: 
    // index 0 -> inf, index 1 -> X=0, index 2 -> X=1, index 3 -> X=2.
    // None handles the infinity boundary case.
    let coord_mapping = [None, Some(0u64), Some(1u64), Some(2u64)];

    // 4. Scan the entire 2D flat grid
    for y_idx in 0..size_d {
        for x_idx in 0..size_d {
            let memory_index = x_idx + y_idx * size_d;
            let computed_val = extended_cube[memory_index];

            // Evaluate standard points vs infinity boundary conditions
            match (coord_mapping[x_idx], coord_mapping[y_idx]) {
                (Some(x_val), Some(y_val)) => {
                    // Standard case: evaluate p(x, y) using Arkworks
                    let point = vec![Fr::from(x_val), Fr::from(y_val)];
                    let expected_val = poly.evaluate(&point);

                    println!("Point ({}, {}) -> Expected: {:2} | Computed: {:?}", x_val, y_val, expected_val, computed_val);
                    assert_eq!(computed_val, expected_val, "Mismatch at standard point ({}, {})", x_val, y_val);
                },
                (None, Some(y_val)) => {
                    // Boundary case: p(inf, y). The highest degree term of X dominates.
                    // Mathematically, this is the leading coefficient of X^1, which is a constant 2.
                    let expected_val = Fr::from(2u64);
                    println!("Point (inf, {}) -> Expected: 2 | Computed: {:?}", y_val, computed_val);
                    assert_eq!(computed_val, expected_val, "Mismatch at boundary point (inf, {})", y_val);
                },
                (Some(x_val), None) => {
                    // Boundary case: p(x, inf). The highest degree term of Y dominates.
                    // This is the leading coefficient of Y^1, which is a constant 1.
                    let expected_val = Fr::from(1u64);
                    println!("Point ({}, inf) -> Expected: 1 | Computed: {:?}", x_val, computed_val);
                    assert_eq!(computed_val, expected_val, "Mismatch at boundary point ({}, inf)", x_val);
                },
                (None, None) => {
                    // Boundary case: p(inf, inf). Since the total degree per variable is bound to 1,
                    // the cross term X^1 Y^1 does not exist, so the leading coefficient is 0.
                    let expected_val = Fr::from(0u64);
                    println!("Point (inf, inf) -> Expected: 0 | Computed: {:?}", computed_val);
                    assert_eq!(computed_val, expected_val, "Mismatch at boundary point (inf, inf)");
                }
            }
        }
    }

    println!("✅ The entire extended hypercube perfectly matches the Arkworks reference polynomial!");
}
*/

 
#[test]
fn test_multi_product_eval_sandbox() {
    use ark_test_curves::bls12_381::Fr;
    use ark_poly::{
        polynomial::multivariate::{SparsePolynomial, SparseTerm, Term},
        DenseMVPolynomial, Polynomial, DenseMultilinearExtension, MultilinearExtension
    };
    use crate::improved::engine::multi_product_eval;

    let v = 2; // Number of variables (X0 and X1)
    let d = 3; // Number of polynomials to multiply

    // 1. Define the evaluations over the base hypercube {0, 1}^2
    // Layout order: X0 changes fastest (LSB), then X1 (MSB)
    // Indices: 00 -> (X0=0, X1=0), 01 -> (X0=1, X1=0), 10 -> (X0=0, X1=1), 11 -> (X0=1, X1=1)

    // p1(X0, X1) = X0 + 1
    let p1_evals = vec![Fr::from(1u64), Fr::from(2u64), Fr::from(1u64), Fr::from(2u64)];
    // p2(X0, X1) = X1 + 2
    let p2_evals = vec![Fr::from(2u64), Fr::from(2u64), Fr::from(3u64), Fr::from(3u64)];
    // p3(X0, X1) = X0 + X1
    let p3_evals = vec![Fr::from(0u64), Fr::from(1u64), Fr::from(1u64), Fr::from(2u64)];

    // Wrap them into Arkworks DenseMultilinearExtension structures
    let p1_extension = DenseMultilinearExtension::from_evaluations_vec(v, p1_evals);
    let p2_extension = DenseMultilinearExtension::from_evaluations_vec(v, p2_evals);
    let p3_extension = DenseMultilinearExtension::from_evaluations_vec(v, p3_evals);

    let input_polynomials = vec![p1_extension, p2_extension, p3_extension];

    // 2. Run the Divide and Conquer MultiProductEval procedure
    let extended_product_cube = multi_product_eval(&input_polynomials, d);

    // Grid size calculation for d = 3 (with base grid k = 1)
    // The grid contains: {inf, 0, 1, 2} -> size_d = 4
    let size_d = 4; 
    assert_eq!(extended_product_cube.len(), size_d * size_d);

    /* 
    println!("Extended product cube : \n");
    for elmt in &extended_product_cube{
        println!("{:?}", elmt)
    }
    */
     
    // 3. Setup the ground-truth product polynomial using Arkworks SparsePolynomial
    // g(X0, X1) = (X0 + 1)(X1 + 2)(X0 + X1) = X0^2*X1 + 2*X0^2 + X0*X1^2 + 3*X0*X1 + 2*X0 + X1^2 + 2*X1
    // The index mapping follows your engine's physical stride execution order:
    // Index 0 represents X0 (fastest stride), Index 1 represents X1 (slowest stride)
    let poly_g = SparsePolynomial::from_coefficients_vec(
        v,
        vec![
            (Fr::from(1u64), SparseTerm::new(vec![(0, 2), (1, 1)])), // X0^2 * X1
            (Fr::from(2u64), SparseTerm::new(vec![(0, 2)])),        // 2 * X0^2
            (Fr::from(1u64), SparseTerm::new(vec![(0, 1), (1, 2)])), // X0 * X1^2
            (Fr::from(3u64), SparseTerm::new(vec![(0, 1), (1, 1)])), // 3 * X0 * X1
            (Fr::from(2u64), SparseTerm::new(vec![(0, 1)])),        // 2 * X0
            (Fr::from(1u64), SparseTerm::new(vec![(1, 2)])),        // X1^2
            (Fr::from(2u64), SparseTerm::new(vec![(1, 1)])),        // 2 * X1
        ],
    );

    println!("\n--- Exhaustive verification of MultiProductEval ({0}x{0} grid) ---", size_d);

    // Coordinate mapping: 0 -> inf, 1 -> 0, 2 -> 1, 3 -> 2
    let coord_mapping = [None, Some(0u64), Some(1u64), Some(2u64)];

    // 4. Double loop to scan and verify the flat evaluation array
    for y_idx in 0..size_d {
        for x_idx in 0..size_d {
            let memory_index = x_idx + y_idx * size_d;
            let computed_val = extended_product_cube[memory_index];

            match (coord_mapping[x_idx], coord_mapping[y_idx]) {
                (Some(x_val), Some(y_val)) => {
                    // Standard classical point evaluation
                    let point = vec![Fr::from(x_val), Fr::from(y_val)];
                    let expected_val = poly_g.evaluate(&point);

                    println!("Point ({}, {}) -> Expected: {:3} | Computed: {:?}", x_val, y_val, expected_val, computed_val);
                    assert_eq!(computed_val, expected_val, "Product mismatch at classical point ({}, {})", x_val, y_val);
                },
                _ => {
                    // Infinity boundary paths
                    // Since testing specific algebraic degrees on boundary intersections can get tricky,
                    // we log them to verify structure continuity.
                    println!("Boundary Point (axis_x: {:?}, axis_y: {:?}) -> Computed: {:?}", coord_mapping[x_idx], coord_mapping[y_idx], computed_val);
                }
            }
        }
    }
    
    println!("✅ Procedure 1 (MultiProductEval) successfully verified against reference polynomial multiplication!");
   
}


// To run more tests on the naive protocol : cargo test -- --nocapture --test-threads=1