// Auxiliary functions

use ark_test_curves::bls12_381::Fr;
use ark_poly::DenseMultilinearExtension;
use ark_ff::{UniformRand, Field, PrimeField};
use ark_linear_sumcheck::ml_sumcheck::data_structures::ListOfProductsOfPolynomials;
use ark_std::rand::Rng;
use ark_std::rc::Rc;
use std::fs::File;

use std::time::Instant;
use std::sync::atomic::Ordering;
use std::io::Write;

use crate::improved::arithmetic::{FAST_PATH_COUNT, SLOW_PATH_COUNT, adaptive_dot_product_accumulate, extrapolate_dot_product}; // Adjust path according to your project structure
use crate::improved::protocol::EvalProductSV;
use crate::improved::streaming::MockStream;

// =================================================================================================
// 1. MULTIVARIATE / PRODUCT OF POLYNOMIALS SETUP (NEW)
// =================================================================================================

/// Generates a list of d independent dense multilinear extensions with random evaluations,
/// along with the Arkworks ListOfProductsOfPolynomials structure required for benchmarking.
/// rng : Random number generator to generate random polynomials p_k (k from 0 to d) in MLE form
/// num_vars : number of variables of each polynomial
/// d : number of multilinear polynomials => maximum degree of g = product(p_0, ..., p_d)
pub fn generate_multivariate_poly_test<R: Rng>(
    rng: &mut R,
    num_vars: usize,
    d: usize,
) -> (
    Vec<DenseMultilinearExtension<Fr>>,
    ListOfProductsOfPolynomials<Fr>,
) {
    // left bitwise shift of num_vars
    // result : 1000...00 (num_vars 0 after the 1) in binary = 2^num_vars in decimal := size of {0,1}^num_vars
    let hypercube_size = 1 << num_vars;

    // Memory reservation : there will be d multilinear polynomials
    let mut list_of_poly = Vec::with_capacity(d);
    let mut poly_rc_vec = Vec::with_capacity(d);

    // 1. Generate d independent dense multilinear extensions with random field elements
    for _ in 0..d {
        // Generate all evaluations of p_k(X) at random from X = 0000...00 (0) to X = 111...11 (2^num_vars := hypercube_size)
        let mut evaluations = Vec::with_capacity(hypercube_size);
        for _ in 0..hypercube_size {
            evaluations.push(Fr::rand(rng));
        }

        // Define the MLE p_k from the evaluations
        let poly = DenseMultilinearExtension::from_evaluations_vec(num_vars, evaluations);

        // Clone for our local custom interactive protocol execution
        list_of_poly.push(poly.clone());
        // Wrap in Rc (Smart pointer) to comply with Arkworks API
        poly_rc_vec.push(Rc::new(poly));
    }

    // 2. Initialize Arkworks product data structure
    let mut list_of_products = ListOfProductsOfPolynomials::new(num_vars);

    // Add the full product p_1 * p_2 * ... * p_d with a generic multiplier coefficient of 1
    list_of_products.add_product(poly_rc_vec, Fr::ONE);

    (list_of_poly, list_of_products)
}

/// Generates a list of d independent dense multilinear extensions whose evaluations
/// on the boolean hypercube are forced to be small integers (Small-Value Setting),
/// along with the Arkworks ListOfProductsOfPolynomials structure required for benchmarking.
pub fn generate_small_value_poly_test<R: Rng>(
    rng: &mut R,
    num_vars: usize,
    d: usize,
) -> (
    Vec<DenseMultilinearExtension<Fr>>,
    ListOfProductsOfPolynomials<Fr>,
) {
    let hypercube_size = 1 << num_vars;
    let mut list_of_poly = Vec::with_capacity(d);
    let mut poly_rc_vec = Vec::with_capacity(d);

    for _ in 0..d {
        let mut evaluations = Vec::with_capacity(hypercube_size);
        for _ in 0..hypercube_size {
            // Force initial evaluations to be small integers (e.g., between 0 and 5)
            // to prevent successive recursive additions/multiplications from overflowing u64
            let small_int = rng.gen_range(0..6) as u64;
            evaluations.push(Fr::from(small_int));
        }

        let poly = DenseMultilinearExtension::from_evaluations_vec(num_vars, evaluations);
        list_of_poly.push(poly.clone());
        poly_rc_vec.push(Rc::new(poly));
    }

    let mut list_of_products = ListOfProductsOfPolynomials::new(num_vars);
    list_of_products.add_product(poly_rc_vec, Fr::ONE);

    (list_of_poly, list_of_products)
}



/// Prints the current state of the adaptive arithmetic counters and resets them to zero.
/// Used for Sanity Check 0 to verify if the fast-path (Small-Big) is triggered.
pub fn print_and_reset_arithmetic_counters() {
    let fast = FAST_PATH_COUNT.swap(0, Ordering::Relaxed);
    let slow = SLOW_PATH_COUNT.swap(0, Ordering::Relaxed);
    let total = fast + slow;

    println!("--------------------------------------------------");
    println!("       SANITY CHECK 0: ARITHMETIC COUNTERS        ");
    println!("--------------------------------------------------");
    println!(" -> Fast-Path Calls (Small-Big Mul): {}", fast);
    println!(" -> Slow-Path Calls (Big-Big Mul)  : {}", slow);
    if total > 0 {
        let ratio = (fast as f64 / total as f64) * 100.0;
        println!(" -> Fast-Path Utilization Rate     : {:.2}%", ratio);
    } else {
        println!(" -> Fast-Path Utilization Rate     : 0.00% (No operations executed)");
    }
    println!("--------------------------------------------------");
}

/// Sanity Check 1: Measures the performance ratio between standard Big-Big multiplication 
/// and our custom optimized Small-Big multiplication.te
/// by forcing 100% Fast-Path vs 100% Slow-Path via controlled slice contents.
/// Sanity Check 1: Measures the performance profile across 4 distinct combinations
/// comparing the legacy adaptive_dot_product_accumulate against the precomputed extrapolate_dot_product
/// using both full-size random fields (Big) and controlled integers (Small).
pub fn run_multiplication_ratio_benchmark() {
    let mut rng = ark_std::test_rng();
    let size = 1_000_000;

    println!("Running Comprehensive Sanity Check 1 Matrix...");

    // 1. Setup inputs
    let big_elements: Vec<Fr> = (0..size).map(|_| Fr::rand(&mut rng)).collect();
    let small_elements: Vec<Fr> = (0..size).map(|i| Fr::from((i % 5 + 1) as u64)).collect();
    let coefficients: Vec<Fr> = (0..size).map(|_| Fr::rand(&mut rng)).collect();
    
    // Precompute limbs required by the new extrapolate function interface
    let coeff_limbs: Vec<_> = coefficients.iter().map(|c| c.into_bigint()).collect();

    // =========================================================================
    // COMBINATION 1: Legacy Adaptive + Big Elements (Pure Slow-Path baseline)
    // =========================================================================
    let mut acc_legacy_big = Fr::ZERO;
    let start_1 = Instant::now();
    adaptive_dot_product_accumulate(&mut acc_legacy_big, &big_elements, &coefficients);
    let dur_legacy_big = start_1.elapsed().as_secs_f64() * 1000.0;

    // =========================================================================
    // COMBINATION 2: Legacy Adaptive + Small Elements (Checks legacy fast-path detection)
    // =========================================================================
    let mut acc_legacy_small = Fr::ZERO;
    let start_2 = Instant::now();
    adaptive_dot_product_accumulate(&mut acc_legacy_small, &small_elements, &coefficients);
    let dur_legacy_small = start_2.elapsed().as_secs_f64() * 1000.0;

    // =========================================================================
    // COMBINATION 3: New Extrapolate + Big Elements (Checks fallback mechanism overhead)
    // =========================================================================
    let mut acc_extrapolate_big = Fr::ZERO;
    let start_3 = Instant::now();
    extrapolate_dot_product(&mut acc_extrapolate_big, &big_elements, &coeff_limbs, &coefficients);
    let dur_extrapolate_big = start_3.elapsed().as_secs_f64() * 1000.0;

    // =========================================================================
    // COMBINATION 4: New Extrapolate + Small Elements (Pure optimized Fast-Path)
    // =========================================================================
    let mut acc_extrapolate_small = Fr::ZERO;
    let start_4 = Instant::now();
    extrapolate_dot_product(&mut acc_extrapolate_small, &small_elements, &coeff_limbs, &coefficients);
    let dur_extrapolate_small = start_4.elapsed().as_secs_f64() * 1000.0;

    // Assert integrity across the execution matrix
    assert_eq!(acc_legacy_big, acc_extrapolate_big, "Mathematical mismatch on Big Elements!");
    assert_eq!(acc_legacy_small, acc_extrapolate_small, "Mathematical mismatch on Small Elements!");

    // Print the benchmark summary directly to the terminal
    println!("------------------------------------------------------------");
    println!("| Configuration                               | Time (ms)  |");
    println!("------------------------------------------------------------");
    println!("| Legacy Adaptive (Big Elements)             | {:10.4} |", dur_legacy_big);
    println!("| Legacy Adaptive (Small Elements)           | {:10.4} |", dur_legacy_small);
    println!("| Extrapolate Precomputed (Big Elements)      | {:10.4} |", dur_extrapolate_big);
    println!("| Extrapolate Precomputed (Small Elements)    | {:10.4} |", dur_extrapolate_small);
    println!("------------------------------------------------------------");

    // Save to the CSV file. Note: The python plotting tool will automatically capture 
    // these 4 discrete categories.
    let mut file = File::create("csv/multiplication_ratio.csv").expect("Unable to create ratio file");
    writeln!(file, "Operation,Time_ms").unwrap();
    writeln!(file, "Legacy (Big Elements),{:.4}", dur_legacy_big).unwrap();
    writeln!(file, "Legacy (Small Elements),{:.4}", dur_legacy_small).unwrap();
    writeln!(file, "Extrapolate (Big Elements),{:.4}", dur_extrapolate_big).unwrap();
    writeln!(file, "Extrapolate (Small Elements),{:.4}", dur_extrapolate_small).unwrap();
    file.flush().unwrap();
}

pub fn bench_offline_seq_vs_parallel(d: usize, l: usize) {
    let mut rng = rand::thread_rng();
    let (list_of_poly, _) = generate_small_value_poly_test(&mut rng, l, d);
    let protocol = EvalProductSV::new(d, l);

    let mut stream = MockStream::new(l, d, &list_of_poly);
    let start = Instant::now();
    let _ = protocol.precomputation_phase(&mut stream); // version parallele actuelle
    println!("d={d} l={l} : parallel offline = {:?}", start.elapsed());

    // Si tu gardes une variante _sequential a cote pour comparaison :
    // let mut stream2 = MockStream::new(l, d, &list_of_poly);
    // let start2 = Instant::now();
    // let _ = protocol.precomputation_phase_sequential(&mut stream2);
    // println!("d={d} l={l} : sequential offline = {:?}", start2.elapsed());
}



// =================================================================================================
// 2. OBSOLETE UNILINEAR FUNCTIONS (COMMENTED OUT FOR FUTURE REFERENCE / BACKUP)
// =================================================================================================
/*

/// Converts a flat index integer into its binary/boolean coordinates over the {0,1}^num_vars hypercube
pub fn i_to_boolean_point(i: usize, num_vars: usize) -> Vec<Fr> {
    let mut n = i;
    let mut point = Vec::with_capacity(num_vars);
    for _ in 0..num_vars {
        point.push(Fr::from((n % 2) as u64));
        n /= 2;
    }
    point
}
    
pub enum PolyType {
    Multilinear,
    Multivariate(usize),
}

pub fn poly_type(poly: &SparsePolynomial<Fr, SparseTerm>) -> PolyType {
    for (_, term) in &poly.terms {
        if term.powers().iter().sum::<usize>() > term.powers().len() {
            return PolyType::Multivariate(poly.degree());
        }
    }
    PolyType::Multilinear
}

pub fn p_i_coeff(
    current_round: usize,
    poly: &SparsePolynomial<Fr, SparseTerm>,
    challenges: &mut Vec<Fr>,
) -> (Fr, Fr) {
    let num_var_h = poly.num_vars - current_round - 1;
    let (a_x, b_x) = find_polynomial_coeff(poly, current_round);
    let num_points = 1 << (num_var_h);
    let mut a = Fr::from(0);
    let mut b = Fr::from(0);
    for i in 0..num_points {
        let x = i_to_boolean_point(i, num_var_h);
        let mut eval_point = challenges.clone();
        eval_point.push(Fr::from(0));
        eval_point.extend(x);
        a += a_x.evaluate(&eval_point);
        b += b_x.evaluate(&eval_point);
    }
    (a, b)
}

pub fn find_polynomial_coeff(
    poly: &SparsePolynomial<Fr, SparseTerm>,
    current_round: usize,
) -> (
    SparsePolynomial<Fr, SparseTerm>,
    SparsePolynomial<Fr, SparseTerm>,
) {
    let mut a_terms: Vec<(Fr, SparseTerm)> = vec![];
    let mut b_terms: Vec<(Fr, SparseTerm)> = vec![];

    for (coeff, term) in &poly.terms {
        if term.iter().any(|(var_index, _power)| *var_index == current_round) {
            let clean_term_vec: Vec<(usize, usize)> = term
                .iter()
                .filter(|(var_index, _)| *var_index != current_round)
                .cloned()
                .collect();
            let term_without_x_i = SparseTerm::new(clean_term_vec);
            a_terms.push((*coeff, term_without_x_i));
        } else {
            b_terms.push((*coeff, term.clone()));
        }
    }

    let a_x = SparsePolynomial::from_coefficients_vec(poly.num_vars, a_terms);
    let b_x = SparsePolynomial::from_coefficients_vec(poly.num_vars, b_terms);
    (a_x, b_x)
}

pub fn generate_sparse_poly<R: Rng>(rng : &mut R, num_monomial : usize) -> SparsePolynomial<Fr, SparseTerm> {
    let n: usize = 10;
    let mut all_monomials: Vec<Vec<usize>> = (0..n).powerset().collect();
    let mut terms = Vec::with_capacity(num_monomial);

    for _ in 0..num_monomial {
        let coeff = Fr::from(rng.gen_range(1..=10));
        let term = if let Some(var_index_vec) = all_monomials.pop() {
            let term_vec: Vec<(usize, usize)> = var_index_vec
                .into_iter()
                .map(|index_var| (index_var, 1))
                .collect();
            SparseTerm::new(term_vec)
        } else {
            continue;
        };
        terms.push((coeff, term));
    }
    SparsePolynomial::from_coefficients_vec(n, terms)
}

pub fn generate_evaluations_from_poly(poly: &SparsePolynomial<Fr, SparseTerm>) -> Vec<Fr> {
    let num_vars = poly.num_vars;
    let total_evals = 2_usize.pow(num_vars as u32);
    let mut evaluations = Vec::with_capacity(total_evals);

    for i in 0..total_evals {
        let mut point = Vec::with_capacity(num_vars);
        for j in 0..num_vars {
            let bit = (i >> j) & 1;
            point.push(Fr::from(bit as u64));
        }
        evaluations.push(poly.evaluate(&point));
    }
    evaluations
}

pub fn generate_small_evaluations_from_poly(poly: &SparsePolynomial<Fr, SparseTerm>) -> Vec<u64> {
    let num_vars = poly.num_vars;
    let total_evals = 2_usize.pow(num_vars as u32);
    let mut evaluations_over_hypercube = Vec::with_capacity(total_evals);

    for i in 0..total_evals {
        let mut point = Vec::with_capacity(num_vars);
        for j in 0..num_vars {
            let bit = (i >> j) & 1;
            point.push(Fr::from(bit as u64));
        }

        let eval_fr = poly.evaluate(&point);
        let eval_big_int = ark_ff::PrimeField::into_bigint(eval_fr);
        let limbs = eval_big_int.as_ref();

        assert!(
            limbs[1..].iter().all(|&limb| limb == 0),
            "Evaluation overflow over first limb!"
        );

        let eval_u64 = limbs[0];
        evaluations_over_hypercube.push(eval_u64);
    }
    evaluations_over_hypercube
}

pub fn generate_poly_test<R: Rng>(rng : &mut R, num_monomial : usize) -> (SparsePolynomial<Fr,SparseTerm> ,ListOfProductsOfPolynomials<Fr>){
    let poly0 = generate_sparse_poly(rng, num_monomial);
    let num_vars = poly0.num_vars;
    let evaluations = generate_evaluations_from_poly(&poly0);
    let poly1 = DenseMultilinearExtension::from_evaluations_vec(num_vars, evaluations);
    let poly_rc = Rc::new(poly1);
    let mut list_of_products = ListOfProductsOfPolynomials::new(num_vars);
    list_of_products.add_product(vec![poly_rc], Fr::from(1u64));
    (poly0, list_of_products)
}

pub fn compute_hypercube_sum(poly: &SparsePolynomial<Fr, SparseTerm>) -> Fr {
    let num_points = 1 << poly.num_vars;
    let mut sum = Fr::from(0);
    for i in 0..num_points {
        let point = i_to_boolean_point(i, poly.num_vars);
        sum += poly.evaluate(&point);
    }
    sum
}

pub fn format_univariate_dense_poly(p_i: &DensePolynomial<Fr>, round: usize) -> String {
    let coeffs = p_i.coeffs();
    let b = coeffs.get(0).cloned().unwrap_or(Fr::from(0));
    let a = coeffs.get(1).cloned().unwrap_or(Fr::from(0));
    format!("p_{}(X_{}) = ({:?}) * X_{} + ({:?})", round, round, a, round, b)
}

pub fn format_multivariate_sparse_poly(poly: &SparsePolynomial<Fr, SparseTerm>) -> String {
    let mut buffer = String::new();
    let _ = write!(buffer, "poly(X) = ");
    for (i, (coeff, term)) in poly.terms.iter().enumerate() {
        if i > 0 { let _ = write!(buffer, " + "); }
        if *coeff != Fr::from(1) {
            let _ = write!(buffer, "{}*", ark_ff::PrimeField::into_bigint(*coeff));
        }
        let vars = term.vars();
        let powers = term.powers();
        for (i, (var, power)) in vars.iter().zip(powers.iter()).enumerate() {
            match power {
                1 => { let _ = write!(buffer, "x_{}", var); },
                _ => { let _ = write!(buffer, "x_{}^{}", var, power); },
            };
            if i < vars.len() - 1 { let _ = write!(buffer, "."); };
        }
    }
    buffer
}
*/
