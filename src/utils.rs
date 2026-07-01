// Auxiliary functions

// Finite field F
use ark_test_curves::bls12_381::Fr;

// Polynomial types
use ark_poly::DenseMultilinearExtension;
use ark_poly::polynomial::multivariate::{SparsePolynomial, SparseTerm, Term};
use ark_poly::univariate::DensePolynomial;
use ark_poly::{DenseMVPolynomial, DenseUVPolynomial, Polynomial};

use ark_ff::{Field, UniformRand};
use ark_linear_sumcheck::ml_sumcheck::data_structures::ListOfProductsOfPolynomials;
use ark_std::rand::Rng;
use ark_std::rc::Rc;

use itertools::Itertools;

// For debugging
use std::fmt::Write;

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
    list_of_products.add_product(poly_rc_vec, Fr::from(1u64));

    (list_of_poly, list_of_products)
}

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

// =================================================================================================
// 2. OBSOLETE UNILINEAR FUNCTIONS (COMMENTED OUT FOR FUTURE REFERENCE / BACKUP)
// =================================================================================================
/*
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
