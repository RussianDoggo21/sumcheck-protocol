use ark_ff::Field;
use ark_test_curves::bls12_381::Fr;

use crate::improved::arithmetic::adaptive_dot_product_accumulate;

pub fn compute_kernel(k: usize) -> Vec<Fr> {
    let mut kernel = Vec::with_capacity(k + 1);
    let target_x = Fr::from(k as u64);

    // 1. Point at infinity: C_inf = k!
    let mut c_inf = Fr::from(1u64);
    for i in 0..k {
        c_inf *= target_x - Fr::from(i as u64);
    }
    kernel.push(c_inf);

    // 2. Classical Lagrange coefficients
    for i in 0..k {
        let mut numerator_prod = Fr::from(1u64);
        let mut denominator_prod = Fr::from(1u64);
        let x_i = Fr::from(i as u64);

        for j in 0..k {
            if j == i { continue; }
            let x_j = Fr::from(j as u64);

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
        let start_idx = 1 + c;
        let end_idx = start_idx + k;

        // Let the adaptive dot product handle the optimization dynamically!
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
        // Dynamic size of coordinates to the +(already size d) and to the right (still size k)
        let left_variants = size_d.pow((j - 1) as u32);
        let right_variants = size_k.pow((num_vars - j) as u32);

        // Allocate the exact size needed for the temporary buffer at step j
        let next_cube_size = left_variants * size_d * right_variants;
        let mut next_cube = vec![Fr::from(0u64); next_cube_size];

        // The stride is the memory distance between two elements along the j-th axis
        let current_axis_stride = right_variants;
        let next_axis_stride = right_variants;

        // Iterate over all slices orthogonal to dimension j (xl, xr)
        for xl in 0..left_variants {
            for xr in 0..right_variants {
                
                // Calculate 1D flattened memory offsets
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