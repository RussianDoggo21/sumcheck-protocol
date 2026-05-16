// Auxilliarry functions

// Finite field F
use ark_test_curves::bls12_381::Fr;

// Polynomial poly
use ark_poly::polynomial::multivariate::{SparsePolynomial, SparseTerm, Term};
use ark_poly::{DenseMVPolynomial, Polynomial};

// Define the type of polynomial for the sumcheck protocol
pub enum PolyType {
    Multilinear,
    Multivariate(usize), // Stores the degree if multivariate
}

// Conversion of an integer in binary
pub fn i_to_boolean_point(i: usize, num_vars: usize) -> Vec<Fr> {
    let mut n = i;
    let mut point = Vec::with_capacity(num_vars);
    for _ in 0..num_vars {
        point.push(Fr::from((n % 2) as u64));
        n /= 2;
    }
    point
}

// Auxilliary function to check if poly is specifically multilinear
pub fn poly_type(poly: &SparsePolynomial<Fr, SparseTerm>) -> PolyType {
    // We check every monomial of poly
    for (_, term) in &poly.terms {
        // If the sum of the powers is greater than the number of variables
        // We have a multivariate polynomial
        // e.g. x_0².x_1 => 2+1 > 2
        if term.powers().iter().sum::<usize>() > term.powers().len() {
            return PolyType::Multivariate(poly.degree());
        }
    }
    // Else, we have a multilinear polynomial
    PolyType::Multilinear
}

pub fn p_i_coeff(
    current_round: usize,
    poly: &SparsePolynomial<Fr, SparseTerm>,
    challenges: &mut Vec<Fr>,
) -> (Fr, Fr) {
    // We have p(X_1, ..., X_i-1, X_i, X_i+1, ..., X_n) = X_i.A(X_1, ..., X_i-1, X_i+1, ..., X_n) + B(X_1, ..., X_i-1, X_i+1, ..., X_n)
    // So p_i(X_i) = a.X_i + b with :
    // a = Sum of evaluation of A(X_1, ..., X_i-1, X_i+1, ..., X_n) over (w_1, ..., w_i-1, x') with x' in H={0,1}^n-i
    // b = Sum of evaluation of B(X_1, ..., X_i-1, X_i+1, ..., X_n) over (w_1, ..., w_i-1, x') with x' in H={0,1}^n-i

    let num_var_h = poly.num_vars - current_round - 1;
    // 1) Determine A(X) and B(X)

    let (a_x, b_x) = find_polynomial_coeff(poly, current_round);

    // 2) Evaluate A and B on the 2^n-i points of H to compute a and b
    let num_points = 1 << (num_var_h);
    let mut a = Fr::from(0);
    let mut b = Fr::from(0);
    for i in 0..num_points {
        let x = i_to_boolean_point(i, num_var_h);
        let mut eval_point = challenges.clone();
        eval_point.extend(x);
        a += a_x.evaluate(&eval_point);
        b += b_x.evaluate(&eval_point);
    }

    // 3) Return a and b
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

    // 1.1) We iterate over all monomials of poly
    for (coeff, term) in &poly.terms {
        // If we find X_i in the monomial, we add it to A(X)
        if term
            .iter()
            .any(|(var_index, _power)| *var_index == current_round)
        {
            // We create a new tuple vector of items (var_index, power)
            // while excluding the one tuple where *var_index == current_round
            let clean_term_vec: Vec<(usize, usize)> = term
                .iter()
                .filter(|(var_index, _)| *var_index != current_round) // We keep ALL BUT current_round
                .cloned() // Clone to create new data
                .collect(); // Putting everything in a vector

            // Initialization of the new term with the precedent vector
            let term_without_x_i = SparseTerm::new(clean_term_vec);

            // We push in A
            a_terms.push((*coeff, term_without_x_i));
        }
        // Else, we add it to B(X)
        else {
            b_terms.push((*coeff, term.clone()));
        }
    }

    // 1.2) We define A and B
    let a_x = SparsePolynomial::from_coefficients_vec(poly.num_vars, a_terms);
    let b_x = SparsePolynomial::from_coefficients_vec(poly.num_vars, b_terms);

    (a_x, b_x)
}

/*
/// Computes the Multilinear Extension (MLE) of a polynomial p_i
/// by evaluating it on every point of the Boolean hypercube.
pub fn mle(p_i: &SparsePolynomial<Fr, SparseTerm>) -> DenseMultilinearExtension<Fr> {
    let num_vars = p_i.num_vars;

    // 1. Calculate the total number of vertices in the hypercube: 2^n
    let num_evals = 1 << num_vars;

    // 2. Pre-allocate the exact capacity to avoid multiple memory reallocations
    let mut evaluations = Vec::with_capacity(num_evals);

    // 3. Iterate through every integer from 0 to 2^n - 1
    for i in 0..num_evals {
        // Convert integer 'i' to a vector of coordinates (0 or 1)
        let point = i_to_boolean_point(i, num_vars);

        // Evaluate the sparse polynomial at this specific vertex
        evaluations.push(p_i.evaluate(&point));
    }

    // 4. Construct the DenseMultilinearExtension from the collected evaluations
    DenseMultilinearExtension::from_evaluations_vec(num_vars, evaluations)
}
*/
