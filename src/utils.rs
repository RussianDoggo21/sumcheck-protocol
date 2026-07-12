// Auxiliary functions

use ark_test_curves::bls12_381::Fr;
use ark_poly::DenseMultilinearExtension;
use ark_ff::{UniformRand, Field};
use ark_linear_sumcheck::ml_sumcheck::data_structures::ListOfProductsOfPolynomials;
use ark_std::rand::Rng;
use ark_std::rc::Rc;

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
