// Auxilliarry functions

// Finite field F
use ark_test_curves::bls12_381::Fr;

// Polynomial poly
use ark_poly::polynomial::multivariate::{SparsePolynomial, SparseTerm, Term};
use ark_poly::{DenseMVPolynomial, DenseUVPolynomial, Polynomial};
use ark_poly::univariate::DensePolynomial;

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

    //println!("Appel à p_i_coeff");

    let num_var_h = poly.num_vars - current_round - 1;
    // 1) Determine A(X) and B(X)

    let (a_x, b_x) = find_polynomial_coeff(poly, current_round);

    // 2) Evaluate A and B on the 2^n-i points of H to compute a and b
    let num_points = 1 << (num_var_h);
    let mut a = Fr::from(0);
    let mut b = Fr::from(0);
    for i in 0..num_points {
        let x = i_to_boolean_point(i, num_var_h);

        // We clone all the challenges onto the evaluation point
        let mut eval_point = challenges.clone();

        // We add a dummy value for the i-th variable
        eval_point.push(Fr::from(0));

        // Finally, we complete by one of the evaluations of the n-i variables over the hypercube
        eval_point.extend(x);
        //println!("eval_point.len = {}, a_x.num_vars = {}, b_x.num_vars = {}", eval_point.len(), a_x.num_vars, b_x.num_vars);
        a += a_x.evaluate(&eval_point);
        b += b_x.evaluate(&eval_point);
    }

    //println!("Fin de p_i_coeff");
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
    //println!("Appel à find_polynomial_coeff");
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

    //println!("Fin de find_polynomial_coeff");
    (a_x, b_x)
    
}

/* ****************************************************************************************************************************************************************** */

/// Print a univariate polynomail g_i(X) in an easy way to read (e.g.: 3*X + 5)
pub fn print_univariate_dense_poly(g_i: &DensePolynomial<Fr>, round: usize) {
    // In the multilinear case, g_i has at most 2 coefficients [b, a] for a*X + b
    let coeffs = g_i.coeffs();
    let b = coeffs.get(0).cloned().unwrap_or(Fr::from(0));
    let a = coeffs.get(1).cloned().unwrap_or(Fr::from(0));
    
    println!("  g_{}(X_{}) = ({:?}) * X_{} + ({:?})", round, round, a, round, b);
}

/// Prints a multivariate sparse polynomial in a readable format (e.g., coeff * x_0^1 * x_2^1 + ...)
pub fn print_multivariate_sparse_poly(poly: &SparsePolynomial<Fr, SparseTerm>) {
    print!("poly(X) = ");
    
    // 1. We use &g_i.terms to borrow the data instead of moving it out of the polynomial.
    // .enumerate() helps us track the index to format the "+" signs beautifully.
    for (i, (coeff, term)) in poly.terms.iter().enumerate() {
        
        // Print the "+" separator between monomials, but not before the first one
        if i > 0 {
            print!(" + ");
        }
        
        // Print the coefficient element from the finite field
        if *coeff != Fr::from(1){
            print!("{:?}*", coeff);
        }
       

        let vars = term.vars();
        let powers = term.powers();
        
        // 2. We use .zip() to safely iterate over variables and their corresponding powers simultaneously
        for (i, (var, power)) in vars.iter().zip(powers.iter()).enumerate() {
            match power {
                1 => print!("x_{}", var),
                _ => print!("x_{}^{}", var, power),
            };
            if i < vars.len() - 1 {
                print!(".");
            };
        }   
    }
    println!(); // Print a newline at the very end
}

pub fn print_sc_poly_and_claim(poly : &SparsePolynomial<Fr, SparseTerm>, gamma : Fr){
    print!("Polynomial to be evaluated : ");
    print_multivariate_sparse_poly(poly);
    println!("Claim : {:?}", gamma);
}

/// Print the details of a round for debugging
pub fn print_round_status(round: usize, claim: Fr, g_i: &DensePolynomial<Fr>, challenges: &mut Vec<Fr>) {
    println!();
    println!("=== DEBUG ROUND {} ===", round + 1);
    println!("  Current Claim V expects: {:?}", claim);
    print!("  Current challenges : ");
    match challenges.len(){
        0 => println!("None (first round)"),
        _ => for (i, challenge) in challenges.iter().enumerate() {
            print!("w_{} = {:?}", i, challenge);
            if i < challenges.len() - 1 {
                print!(", ");
            }
        },
    }
    println!();
    print_univariate_dense_poly(g_i, round);
    
    let eval_0 = g_i.evaluate(&Fr::from(0));
    let eval_1 = g_i.evaluate(&Fr::from(1));
    println!("  g_{}(0) = {:?}", round, eval_0);
    println!("  g_{}(1) = {:?}", round, eval_1);
    println!("  g_{}(0) + g_{}(1) = {:?}", round, round, eval_0 + eval_1);
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
