// Auxiliary functions

use ark_test_curves::bls12_381::Fr;
use ark_poly::polynomial::multivariate::{SparsePolynomial, SparseTerm, Term};
use ark_poly::univariate::DensePolynomial;
use ark_poly::{DenseMultilinearExtension, DenseMVPolynomial, DenseUVPolynomial, Polynomial};
use ark_ff::{UniformRand, Field};
use ark_linear_sumcheck::ml_sumcheck::data_structures::ListOfProductsOfPolynomials;
use ark_std::rand::Rng;
use ark_std::rc::Rc;
use itertools::Itertools;

// For debugging
use std::fmt::Write;

// =================================================================================================
// 1. MULTIVARIATE / PRODUCT OF POLYNOMIALS SETUP (NEW)
// =================================================================================================

pub fn generate_multivariate_poly_test<R: Rng>(
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
            evaluations.push(Fr::rand(rng));
        }
        let poly = DenseMultilinearExtension::from_evaluations_vec(num_vars, evaluations);
        list_of_poly.push(poly.clone());
        poly_rc_vec.push(Rc::new(poly));
    }

    let mut list_of_products = ListOfProductsOfPolynomials::new(num_vars);
    list_of_products.add_product(poly_rc_vec, Fr::ONE);

    (list_of_poly, list_of_products)
}

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

pub enum PolyType {
    Multilinear,
    Multivariate(usize),
}

pub fn i_to_boolean_point(i: usize, num_vars: usize) -> Vec<Fr> {
    let mut n = i;
    let mut point = Vec::with_capacity(num_vars);
    for _ in 0..num_vars {
        point.push(Fr::from((n % 2) as u64));
        n /= 2;
    }
    point
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
    let mut a = Fr::ZERO;
    let mut b = Fr::ZERO;
    for i in 0..num_points {
        let x = i_to_boolean_point(i, num_var_h);
        let mut eval_point = challenges.clone();
        eval_point.push(Fr::ZERO);
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
        if term
            .iter()
            .any(|(var_index, _power)| *var_index == current_round)
        {
            let clean_term_vec: Vec<(usize, usize)> = term
                .iter()
                .filter(|(var_index, _)| *var_index != current_round)
                .cloned()
                .collect();
            let term_without_x_i = SparseTerm::new(clean_term_vec);
            a_terms.push((*coeff, term_without_x_i));
        }
        else {
            b_terms.push((*coeff, term.clone()));
        }
    }

    let a_x = SparsePolynomial::from_coefficients_vec(poly.num_vars, a_terms);
    let b_x = SparsePolynomial::from_coefficients_vec(poly.num_vars, b_terms);

    (a_x, b_x)
}

pub fn generate_sparse_poly<R: Rng>(rng : &mut R) -> SparsePolynomial<Fr, SparseTerm>{
    let n: usize = rng.gen_range(2..=10);
    let mut all_monomials: Vec<Vec<usize>> = (0..n).powerset().collect();
    let num_monomial = rng.gen_range(2..=10);
    let mut terms = Vec::with_capacity(num_monomial);

    for _ in 1..num_monomial {
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

/// NEW ! TO UNDERSTAND : factored to reuse generate_evaluations_from_poly instead of
/// duplicating the hypercube/boolean-point iteration logic -- computes the Fr
/// evaluations once, then maps each down to its raw u64 low limb, rather than
/// re-walking the hypercube a second time with near-identical code.
pub fn generate_small_evaluations_from_poly(poly: &SparsePolynomial<Fr, SparseTerm>) -> Vec<u64> {
    generate_evaluations_from_poly(poly)
        .into_iter()
        .map(|eval_fr| ark_ff::PrimeField::into_bigint(eval_fr).as_ref()[0])
        .collect()
}

pub fn generate_poly_test<R: Rng>(rng : &mut R) -> (SparsePolynomial<Fr,SparseTerm> ,ListOfProductsOfPolynomials<Fr>){
    let poly0 = generate_sparse_poly(rng);
    let num_vars = poly0.num_vars;
    let evaluations = generate_evaluations_from_poly(&poly0);
    let poly1 = DenseMultilinearExtension::from_evaluations_vec(num_vars, evaluations);
    let poly_rc = Rc::new(poly1);
    let mut list_of_products = ListOfProductsOfPolynomials::new(num_vars);
    list_of_products.add_product(
        vec![poly_rc],
        Fr::ONE
    );
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
    let b = coeffs.get(0).cloned().unwrap_or(Fr::ZERO);
    let a = coeffs.get(1).cloned().unwrap_or(Fr::ZERO);

    format!(
        "p_{}(X_{}) = ({:?}) * X_{} + ({:?})",
        round, round, a, round, b
    )
}

pub fn format_multivariate_sparse_poly(poly: &SparsePolynomial<Fr, SparseTerm>) -> String {
    let mut buffer = String::new();
    let _ = write!(buffer, "poly(X) = ");

    for (i, (coeff, term)) in poly.terms.iter().enumerate() {
        if i > 0 {
           let _ = write!(buffer, " + ");
        }
        if *coeff != Fr::ONE {
            let _ = write!(buffer, "{}*", ark_ff::PrimeField::into_bigint(*coeff));
        }

        let vars = term.vars();
        let powers = term.powers();

        for (i, (var, power)) in vars.iter().zip(powers.iter()).enumerate() {
            match power {
                1 => {
                    let _ = write!(buffer, "x_{}", var);
                },
                _ => {
                    let _ = write!(buffer, "x_{}^{}", var, power);
                },
            };
            if i < vars.len() - 1 {
                let _ = write!(buffer, ".");
            };
        }
    }

    buffer
}