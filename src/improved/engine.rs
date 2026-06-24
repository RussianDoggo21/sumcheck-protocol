use ark_ff::Field;
use ark_test_curves::bls12_381::Fr;
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};

use crate::improved::arithmetic::adaptive_dot_product_accumulate;

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
            if j == i { continue; }
            let x_j = x_points[j];

            // Computation of the numerator and the denominator factor by factor
            numerator_prod *= target_x - x_j; // k-j
            denominator_prod *= x_i - x_j; // i-j
        }

        // Single modular inverse per Lagrange coefficient
        let lagrange_coeff = numerator_prod * denominator_prod.inverse().expect("Unexpected zero denominator");
        kernel.push(lagrange_coeff);
    }

    kernel
}


/// Extrapolates a vector of evaluations using an ultra-optimized sliding window approach.
/// - `evals_`:  initially contains e_k (size k+1): [p(inf), p(0), p(1), ..., p(k-1)]
/// - `kernel`: precomputed kernel slice of size k+1, containing [C_inf, C_0, ..., C_{k-1}]
/// - `k`: Base size of the window
/// - `num_extrap`: number of additional points to calculate (e.g., 4 to grow from e_4 to e_8)
pub fn univariate_extrapolate(
    evals: &mut Vec<Fr>, 
    kernel: &[Fr], 
    k: usize, 
    num_extrap: usize
) {
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
        adaptive_dot_product_accumulate(&mut next_val, &evals[start_idx..end_idx], kernel_classical);
        
        evals.push(next_val);
    }
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

    // 1. Compute our Lagrange kernel ONCE at the very beginning
    // since we will always extrapolate from U_k to U_d
    let kernel = compute_kernel(k);

    // Initialize our working hypercube with the input data (U_k^v)
    let mut current_cube = initial_evals.to_vec();

    // 2. Main Loop: Loop over each dimension j from 1 to v
    for j in 1..=num_vars {
        // Dynamic size of coordinates to the left(already size d) and to the right (still size k)
        // left_variants := #U_d^(j-1) = (d+1)^(j-1)
        // right_variants := #U_k^(v-j) = (k+1)^(v-j)
        let left_variants = size_d.pow((j - 1) as u32);
        let right_variants = size_k.pow((num_vars - j) as u32);

        // Allocate the exact size needed for the temporary buffer at step j
        // The j-th dimension grows from (k+1) elements (U_k) to (d+1) elements (U_d)
        let next_cube_size = left_variants * size_d * right_variants; // #U_d^(j-1) * (d+1) * #U_k^(v-j)
        let mut next_cube = vec![Fr::ZERO; next_cube_size];

        // The stride is the memory distance between two elements along the j-th axis
        // BLACK-BOXED
        let current_axis_stride = right_variants;
        let next_axis_stride = right_variants;

        // Iterate over all slices orthogonal to dimension j (xl, xr)
        for xl in 0..left_variants {
            for xr in 0..right_variants {
                
                // Calculate 1D flattened memory offsets
                // BLACK-BOXED
                let current_line_offset = xl * size_k * right_variants + xr;
                let next_line_offset = xl * size_d * right_variants + xr;

                // Extraction: Get the univariate line of size (k+1) along axis j
                let mut univariate_slice = Vec::with_capacity(size_d);
                for idx in 0..size_k {
                    let memory_index = current_line_offset + idx * current_axis_stride;
                    univariate_slice.push(current_cube[memory_index]);
                }

                // Core execution: Extrapolate this line to size (d+1) using our adaptive window
                univariate_extrapolate(&mut univariate_slice, &kernel, k, num_extrap);

                // Injection: Write the expanded line back into the temporary next layout
                for idx in 0..size_d {
                    let memory_index = next_line_offset + idx * next_axis_stride;
                    next_cube[memory_index] = univariate_slice[idx];
                }
            }
        }

        // The updated layout configuration becomes the reference for the next dimension
        current_cube = next_cube;
    }

    // After v rounds, current_cube matches U_d^v and has size (d+1)^v
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

    assert_eq!(polynomials.len(), d, "The number of polynomials provided must match d");
    assert!(d > 0, "Cannot compute the product of an empty slice of polynomials");

    let expected_len = 1 << v; //#({0,1}^v) = 2^v
    for (i, poly) in polynomials.iter().enumerate() {
        assert_eq!(
            poly.len(), 
            expected_len, 
            "Polynomial at index {} does not have the expected size 2^{} = {}", 
            i, v, expected_len
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
                    let p0 = current_grid[offset + i];          // Evaluation at point 0
                    let p1 = current_grid[offset + stride + i]; // Evaluation at point 1
                    let p_inf = p1 - p0;                        // Projective limit at infinity

                    // Reorder elements into the target layout based explicitly on U_1 domain
                    for (u_idx, u_point) in u_1_domain.iter().enumerate() {
                        match u_point {
                            EvaluationPoint::Infinity => {
                                next_grid[offset + u_idx * stride + i] = p_inf;
                            },
                            EvaluationPoint::Value(0) => {
                                next_grid[offset + u_idx * stride + i] = p0;
                            },
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
    let m = d/2;
    
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
    assert_eq!(q_l_prime.len(), q_r_prime.len(), "Size mismatch during pointwise multiplication");

    // 4. Combine: Pointwise product of evaluations (Hadamard product)
    let mut g = Vec::with_capacity(q_l_prime.len());
    for i in 0..q_l_prime.len() {
        g.push(q_l_prime[i] * q_r_prime[i]);
    }

    //println!("Debugging of multi_product_eval : end");

    g
}