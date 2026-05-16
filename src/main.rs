// First implementation of sumcheck protocol using arkworks

// Finite field F
use ark_std::UniformRand;
use ark_test_curves::bls12_381::Fr;

// Polynomial poly
use ark_poly::polynomial::multivariate::{SparsePolynomial, SparseTerm, Term};
use ark_poly::univariate::DensePolynomial;
use ark_poly::{DenseMVPolynomial, DenseUVPolynomial, Polynomial};

// Define the type of polynomial for the sumcheck protocol
enum PolyType {
    Multilinear,
    Multivariate(usize), // Stores the degree if multivariate
}

fn main() {
    // poly(x_0, x_1, x_2) = 2*x_0 + x_0*x_2 + x_1*x_2
    let poly = SparsePolynomial::from_coefficients_vec(
        3,
        vec![
            (Fr::from(2), SparseTerm::new(vec![(0, 1)])),
            (Fr::from(1), SparseTerm::new(vec![(0, 1), (2, 1)])),
            (Fr::from(1), SparseTerm::new(vec![(1, 1), (2, 1)])),
            (Fr::from(0), SparseTerm::new(vec![])),
        ],
    );
    let gamma = Fr::from(10);
    sc_protocol(&poly, gamma);
}

// Sumcheck protocol
fn sc_protocol(poly: &SparsePolynomial<Fr, SparseTerm>, gamma: Fr) -> bool {
    // 1. Test if p is multilinear (easy case) or multivariate (general case)
    let poly_type = poly_type(poly);

    // Do the actual sumcheck protocol while precising the type of poly
    let mut challenges = vec![];
    let mut current_claim = gamma;
    for round in 0..poly.num_vars {
        let check_round_i =
            sc_protocol_round(round, poly, &poly_type, &mut current_claim, &mut challenges);
        if check_round_i == false {
            return false;
        }
    }
    true
}

// i-th round of the sumcheck protocol
// poly is a multivariate polynomial which can be decomposed into a product of d multilinear polynomials
// For simplicity, we start on the case where poly is a simple multilinear polynomial ??
fn sc_protocol_round(
    current_round: usize,
    poly: &SparsePolynomial<Fr, SparseTerm>,
    poly_type: &PolyType,
    current_claim: &mut Fr,
    challenges: &mut Vec<Fr>,
) -> bool {
    println!("Starting round {}", current_round + 1);

    // 1) P generates the MLE g_i(X) of the current univariate polynomial p_i(X_i) = SUM_ON_ALPHAS(poly(alpha_1, ..., X_i, ... alpha_n))
    // g_i = mle(p_i)
    let g_i: DensePolynomial<Fr> = prover_i(current_round, poly, poly_type, challenges);

    // 2.1) V checks that Sum of g_i(X) over {0,1} is the current_claim (i.e g_i(0) + g_i(1) = current_claim)
    // 2.2) If that's the case, V sends a new challenge
    let w_i = match verifier_i(&g_i, *current_claim) {
        Ok(w_i) => w_i,
        Err(e) => {
            println!("{}", e);
            return false;
        }
    };
    challenges.push(w_i);

    // 3) Next round
    // 3.1) We define the next claim
    *current_claim = g_i.evaluate(&w_i);
    // 3.2) We also confirm that the verifier accepted the proof of the round
    true
}

// Prover algorithm at the i-th round
fn prover_i(
    current_round: usize,
    poly: &SparsePolynomial<Fr, SparseTerm>,
    poly_type: &PolyType,
    challenges: &mut Vec<Fr>,
) -> DensePolynomial<Fr> {
    let result = match poly_type {
        PolyType::Multilinear => prover_i_multilinear(current_round, poly, challenges),
        PolyType::Multivariate(_d) => prover_i_multivariate(current_round, poly, challenges),
    };
    result
}

fn prover_i_multilinear(
    current_round: usize,
    poly: &SparsePolynomial<Fr, SparseTerm>,
    challenges: &mut Vec<Fr>,
) -> DensePolynomial<Fr> {
    // 1) P computes the dense univariate polynomial p_i(X) defined on Fr[X]

    // 1.1) Computation of the coefficients a and b such that p_i(X) = a.X + b
    let (a, b) = p_i_coeff(current_round, poly, challenges);
    // 1.2) Generation of p_i(X)
    let p_i = DensePolynomial::from_coefficients_vec(vec![b, a]);
    p_i
}

/// A METTRE A JOUR
fn prover_i_multivariate(
    current_round: usize,
    poly: &SparsePolynomial<Fr, SparseTerm>,
    challenges: &mut Vec<Fr>,
) -> DensePolynomial<Fr> {
    todo!("Le cas général multivarié n'est pas encore implémenté !");
}

// Verifier algorithm at the i-th round

fn verifier_i(g_i: &DensePolynomial<Fr>, current_claim: Fr) -> Result<Fr, &'static str> {
    let eval_0 = g_i.evaluate(&Fr::from(0));
    let eval_1 = g_i.evaluate(&Fr::from(1));
    let check_sum = eval_0 + eval_1;

    // If the check fails, we stop the programm : the Prover is cheating
    if check_sum != current_claim {
        return Err("Sumcheck verification failed: g_i(0) + g_i(1) != current_claim");
    };
    // If the check pass, V "sends" a random field element w_i
    let mut rng = ark_std::test_rng();
    let w_i = Fr::rand(&mut rng);
    Ok(w_i)
}

// Conversion of an integer in binary
fn i_to_boolean_point(i: usize, num_vars: usize) -> Vec<Fr> {
    let mut n = i;
    let mut point = Vec::with_capacity(num_vars);
    for _ in 0..num_vars {
        point.push(Fr::from((n % 2) as u64));
        n /= 2;
    }
    point
}

// Auxilliary function to check if poly is specifically multilinear
fn poly_type(poly: &SparsePolynomial<Fr, SparseTerm>) -> PolyType {
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

fn p_i_coeff(
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

fn find_polynomial_coeff(
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
fn mle(p_i: &SparsePolynomial<Fr, SparseTerm>) -> DenseMultilinearExtension<Fr> {
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
