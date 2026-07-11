use ark_ff::{Field, PrimeField,BigInteger256};
use ark_test_curves::bls12_381::Fr;

use crate::improved::arithmetic::extrapolate_dot_product;

use rayon::prelude::*;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum EvaluationPoint {
    Infinity,
    Value(u64),
}

/// Generates the full extended grid domain U_d = { inf, 0, 1, ..., d-1 }
/// Used for indexing multi_product_eval as well as univariate_extrapolation and multivariate extrapolation.
pub fn get_u_domain(d: usize) -> Vec<EvaluationPoint> {
    let mut domain = Vec::with_capacity(d + 1);
    domain.push(EvaluationPoint::Infinity);
    for val in 0..d {
        domain.push(EvaluationPoint::Value(val as u64));
    }
    domain
}

/// Generates the restricted domain U_d_hat = { inf, 1, ..., d-1 }
/// Used specifically for the round polynomials s_i(u) in LinearTime_SC.
pub fn get_u_hat_domain(d: usize) -> Vec<EvaluationPoint> {
    let mut domain = Vec::with_capacity(d);
    domain.push(EvaluationPoint::Infinity);
    for val in 1..d {
        domain.push(EvaluationPoint::Value(val as u64));
    }
    domain
}

pub fn compute_kernel(k: usize) -> Vec<Fr> {
    let mut kernel = Vec::with_capacity(k + 1);
    let target_x = Fr::from(k as u64);

    // Used later in the for loops
    // Allows to reduce the number of conversion from u64 to Fr
    let x_points: Vec<Fr> = (0..k).map(|i| Fr::from(i as u64)).collect();

    // 1. Point at infinity: C_inf = k!
    let mut c_inf = Fr::ONE;
    for i in 0..k {
        c_inf *= target_x - x_points[i];
    }
    kernel.push(c_inf);

    // 2. Classical Lagrange coefficients
    for i in 0..k {
        let mut numerator_prod = Fr::ONE;
        let mut denominator_prod = Fr::ONE;
        let x_i = x_points[i];

        for j in 0..k {
            if j == i {
                continue;
            }
            let x_j = x_points[j];

            // Computation of the numerator and the denominator factor by factor
            numerator_prod *= target_x - x_j; // k-j
            denominator_prod *= x_i - x_j; // i-j
        }

        // Single modular inverse per Lagrange coefficient
        let lagrange_coeff = numerator_prod
            * denominator_prod
                .inverse()
                .expect("Unexpected zero denominator");
        kernel.push(lagrange_coeff);
    }

    kernel
}

/// Performs multivariate polynomial extrapolation from U_k^v to U_d^v.
///
/// - `initial_evals`: Flattened evaluations over U_k^v (size must be (k+1)^num_vars)
/// - `k`: Degree parameter / initial window size
/// - `num_extrap`: Number of extra points to compute per axis (d - k)
/// - `num_vars`: Number of variables (v)
pub fn multivariate_extrapolate(
    initial_evals: &[Fr],
    k: usize,
    num_extrap: usize,
    num_vars: usize,
) -> Vec<Fr> {
    let size_k = k + 1;
    let size_d = k + num_extrap + 1;

    assert_eq!(
        initial_evals.len(),
        size_k.pow(num_vars as u32),
        "Initial evaluations vector size does not match (k+1)^v"
    );

    let kernel = compute_kernel(k);
    let mut current_cube = initial_evals.to_vec();

    // Global retrieving of the coefficients to avoid repeated deferencement
    let kernel_inf = kernel[0];
    let kernel_classical = &kernel[1..];
    let kernel_classical_bigints: Vec<BigInteger256> = kernel_classical.iter().map(|c| c.into_bigint()).collect();

    for j in 1..=num_vars {
        let left_variants = size_d.pow((j - 1) as u32);
        let right_variants = size_k.pow((num_vars - j) as u32);

        let next_cube_size = left_variants * size_d * right_variants;
        let mut next_cube = vec![Fr::ZERO; next_cube_size];

        let current_axis_stride = right_variants;
        let next_axis_stride = right_variants;

        // UNIQUE WORK BUFFER BY AXIS (prevent millions of Vec allocations)
        let mut working_row = vec![Fr::ZERO; size_d];

        for xl in 0..left_variants {
            for xr in 0..right_variants {
                let current_line_offset = xl * size_k * right_variants + xr;
                let next_line_offset = xl * size_d * right_variants + xr;

                // 1. Direct copy from current_cube to our static buffer
                for idx in 0..size_k {
                    let memory_index = current_line_offset + idx * current_axis_stride;
                    working_row[idx] = current_cube[memory_index];
                }

                // 2. Inlined, ultra-fast equivalent of univariate_extrapolate
                let inf_term = working_row[0] * kernel_inf;
                
                for c in 0..num_extrap {
                    let mut next_val = inf_term;
                    let start_idx = 1 + c;
                    let end_idx = start_idx + k;

                    // Direct call on our memory pre-allocated buffer
                    extrapolate_dot_product(&mut next_val, &working_row[start_idx..end_idx], &kernel_classical_bigints, kernel_classical);

                    working_row[size_k + c] = next_val;
                }

                // 3. Direct injection from our static buffer into next_cube
                for idx in 0..size_d {
                    let memory_index = next_line_offset + idx * next_axis_stride;
                    next_cube[memory_index] = working_row[idx];
                }
            }
        }
        current_cube = next_cube;
    }

    current_cube
}

/// Implementation of Procedure 1: MultiProductEval
/// Recursively computes the evaluations of g(x) = \prod_{i=1}^d p_i(x) over U_d^v
///
/// # Arguments
/// * `polynomials` - A slice of vectors where each Vec<Fr> contains the evaluations of a polynomial over {0,1}^v
/// * `d` - The total number of polynomials to multiply
/// * `v` - The number of variables for the polynomials in this current sub-cube
pub fn multi_product_eval(polynomials: &[Vec<Fr>], d: usize, v: usize) -> Vec<Fr> {
    assert_eq!(
        polynomials.len(),
        d,
        "The number of polynomials provided must match d"
    );
    assert!(
        d > 0,
        "Cannot compute the product of an empty slice of polynomials"
    );

    let expected_len = 1 << v; //#({0,1}^v) = 2^v
    for (i, poly) in polynomials.iter().enumerate() {
        assert_eq!(
            poly.len(),
            expected_len,
            "Polynomial at index {} does not have the expected size 2^{} = {}",
            i,
            v,
            expected_len
        );
    }

    // 1. Base case: if d = 1, g(x) = p_1(x)
    // Map the initial Boolean hypercube {0, 1}^v to the U_1^v grid layout [inf, 0]
    if d == 1 {
        let mut current_grid = polynomials[0].clone();
        let u_1_domain = get_u_domain(1); // Contains [Infinity, Value(0)]

        for var in 0..v {
            // BLACK-BOXED
            let stride = usize::pow(2, var as u32);
            let chunk_size = stride * 2;

            let mut next_grid = vec![Fr::ZERO; current_grid.len()];

            for chunk in 0..(current_grid.len() / chunk_size) {
                let offset = chunk * chunk_size; // BLACK-BOXED
                for i in 0..stride {
                    let p0 = current_grid[offset + i]; // Evaluation at point 0
                    let p1 = current_grid[offset + stride + i]; // Evaluation at point 1
                    let p_inf = p1 - p0; // Projective limit at infinity

                    // Reorder elements into the target layout based explicitly on U_1 domain
                    for (u_idx, u_point) in u_1_domain.iter().enumerate() {
                        match u_point {
                            EvaluationPoint::Infinity => {
                                next_grid[offset + u_idx * stride + i] = p_inf;
                            }
                            EvaluationPoint::Value(0) => {
                                next_grid[offset + u_idx * stride + i] = p0;
                            }
                            _ => unreachable!("U_1 domain should only contain Inf and 0"),
                        }
                    }
                }
            }
            current_grid = next_grid;
        }
        return current_grid;
    }

    // 2. Divide: split the polynomials into two halves
    let m = d / 2;

    // Recursive calls for left and right sub-products
    let q_l = multi_product_eval(&polynomials[0..m], m, v);
    let q_r = multi_product_eval(&polynomials[m..d], d - m, v);

    // 3. Extrapolate both halves to the target domain U_d^v
    // For q_l: currently on U_m^v, needs to reach U_d^v.
    // Number of classical points to add = d - m
    let num_extrap_l = d - m;
    let q_l_prime = multivariate_extrapolate(&q_l, m, num_extrap_l, v);

    // For q_r: currently on U_{d-m}^v, needs to reach U_d^v.
    // Number of classical points to add = m
    let num_extrap_r = m;
    let q_r_prime = multivariate_extrapolate(&q_r, d - m, num_extrap_r, v);

    // Sanity check: both extended cubes must have identical sizes
    assert_eq!(
        q_l_prime.len(),
        q_r_prime.len(),
        "Size mismatch during pointwise multiplication"
    );

    // 4. Combine: Pointwise product of evaluations (Hadamard product)
    let mut g = Vec::with_capacity(q_l_prime.len());
    for i in 0..q_l_prime.len() {
        g.push(q_l_prime[i] * q_r_prime[i]);
    }

    g
}

/// Sequential bookkeeping reduction. Takes a flat subcube chunk of size 2^w_1 and
/// iteratively folds it down to a single point by applying the window's challenges (r1, ..., r_w1).
pub fn fold_hypercube_chunk(chunk: &[Fr], challenges: &[Fr]) -> Fr {
    let omega_1 = challenges.len();
    assert_eq!(
        chunk.len(),
        1 << omega_1,
        "Chunk size mismatch with window size"
    );

    let mut working_buffer = chunk.to_vec();

    for round in 0..omega_1 {
        let current_size = 1 << (omega_1 - round);
        let next_size = current_size / 2;
        let r = challenges[round];

        for idx in 0..next_size {
            let p0 = working_buffer[idx << 1];
            let p1 = working_buffer[(idx << 1) | 1];
            // Linear combination over the boolean coordinate
            working_buffer[idx] = p0 + r * (p1 - p0);
        }
    }
    working_buffer[0]
}


/// Performs the infinity-aware dynamic folding step over the evaluation grid
#[inline(never)]
pub fn dynamic_folding_step<F: Field>(
    q: &[F],
    challenge: F,
    d: usize,
    base: usize,
    remaining_vars: usize,
) -> Vec<F> {
    let next_grid_size = usize::pow(base, remaining_vars as u32);
    let mut next_q = vec![F::zero(); next_grid_size];

    // --- PHASE 1: Pre-compute scalars for the unique round challenge ---
    
    // Finite evaluation points are {0, 1, ..., d-1} (size d)
    let mut classical_points = vec![F::zero(); d];
    for val in 0..d {
        classical_points[val] = F::from(val as u64);
    }

    // 1.(a) Pre-compute the classical Lagrange basis coefficients for finite points
    let mut finite_lagrange_coeffs = vec![F::zero(); d];
    for i in 0..d {
        let mut numerator = F::one();
        let mut denominator = F::one();
        let x_i = classical_points[i];
        
        for j in 0..d {
            if i != j {
                let x_j = classical_points[j];
                numerator *= challenge - x_j;
                denominator *= x_i - x_j;
            }
        }
        finite_lagrange_coeffs[i] = numerator * denominator.inverse().unwrap_or(F::zero());
    }

    // 1.(b) Pre-compute the vanishing polynomial product: \prod_{k=0}^{d-1} (challenge - x_k)
    let mut vanishing_prod = F::one();
    for &x_k in classical_points.iter() {
        vanishing_prod *= challenge - x_k;
    }

    // --- PHASE 2: Critical cross-dimension dot product loop (O(next_grid_size * d)) ---
    for future_idx in 0..next_grid_size {
        // Extract the evaluation at infinity (index 0 of current variable stride)
        let old_grid_idx_inf = future_idx * base;
        let leading_coeff = q[old_grid_idx_inf];

        // Part A from Lemma 2: leading_coeff * vanishing_prod
        let mut interpolated_val = leading_coeff * vanishing_prod;

        // Part B from Lemma 2: \sum_{i=0}^{d-1} evals[i + 1] * finite_lagrange_coeffs[i]
        for i in 0..d {
            let old_grid_idx_finite = (i + 1) + future_idx * base;
            interpolated_val += q[old_grid_idx_finite] * finite_lagrange_coeffs[i];
        }

        next_q[future_idx] = interpolated_val;
    }

    next_q
}

/* 
/// Extrapolates a vector of evaluations using an ultra-optimized sliding window approach.
/// - `evals_`:  initially contains e_k (size k+1): [p(inf), p(0), p(1), ..., p(k-1)]
/// - `kernel`: precomputed kernel slice of size k+1, containing [C_inf, C_0, ..., C_{k-1}]
/// - `k`: Base size of the window
/// - `num_extrap`: number of additional points to calculate (e.g., 4 to grow from e_4 to e_8)
pub fn univariate_extrapolate(evals: &mut Vec<Fr>, kernel: &[Fr], k: usize, num_extrap: usize) {
    evals.reserve(num_extrap);

    let kernel_inf = kernel[0];
    let kernel_classical = &kernel[1..];

    let p_inf = evals[0];
    let inf_term = p_inf * kernel_inf;

    for c in 0..num_extrap {
        // Target initialized with the infinity term
        let mut next_val = inf_term;

        // Extract the sub-slice of Fr elements currently inside our shifting window
        // &evals[start_idx..end_idx] := p(c), ..., p(c+k-1)
        let start_idx = 1 + c;
        let end_idx = start_idx + k;

        // next_val += dot_product(evals[strat_idx..end_idx], kernel_classical)
        adaptive_dot_product_accumulate(
            &mut next_val,
            &evals[start_idx..end_idx],
            kernel_classical,
        );

        evals.push(next_val);
    }
}
*/

/* 
/// Linearly interpolates a set of d+1 evaluations over U_d at a specific challenge point `challenge`.
/// Expected input layout structure: [p(inf), p(0), p(1), ..., p(d-1)]
pub fn interpolate_at_point(evals: &[Fr], challenge: Fr) -> Fr {
    let d = evals.len() - 1;
    let mut classical_points = Vec::with_capacity(d);
    for val in 0..d {
        classical_points.push(Fr::from(val as u64));
    }

    // 1. Classical Lagrange interpolation over the d finite points (0 to d-1)
    let mut lagrange_sum = Fr::ZERO;
    for i in 0..d {
        let mut numerator = Fr::ONE;
        let mut denominator = Fr::ONE;
        let x_i = classical_points[i];

        for j in 0..d {
            if i != j {
                let x_j = classical_points[j];
                numerator *= challenge - x_j;
                denominator *= x_i - x_j;
            }
        }
        let l_i = numerator * denominator.inverse().unwrap_or(Fr::ZERO);
        lagrange_sum += evals[i + 1] * l_i; // evals[i+1] corresponds to evaluation of the poly at point 'i'
    }

    // 2. Vanishing polynomial product: \prod (r_i - x_k)
    let mut vanishing_prod = Fr::ONE;
    for x_i in &classical_points {
        vanishing_prod *= challenge - x_i;
    }

    // 3. Extract the leading coefficient (evaluation at infinity at index 0)
    let leading_coeff = evals[0];

    // Combine using Lemma 2 formula
    (leading_coeff * vanishing_prod) + lagrange_sum
}
*/