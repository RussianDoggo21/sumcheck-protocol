use ark_ff::Field;
use ark_test_curves::bls12_381::Fr;
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};

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

/// Implementation of Procedure 1: MultiProductEval
/// Recursively computes the evaluations of g(x) = \prod_{i=1}^d p_i(x) over U_d^v
///
/// # Arguments
/// * `polynomials` - A slice of DenseMultilinearExtension representing multilinear polynomials over {0,1}^v
/// * `d` - The total number of polynomials to multiply
pub fn multi_product_eval(polynomials: &[DenseMultilinearExtension<Fr>], d: usize) -> Vec<Fr> {

    //println!("\nDebugging of multi_product_eval : beginning");

    assert_eq!(polynomials.len(), d, "The number of polynomials provided must match d");
    assert!(d > 0, "Cannot compute the product of an empty slice of polynomials");

    // Check that all input polynomials share the same number of variables
    let v = polynomials[0].num_vars();
    for i in 1..polynomials.len(){
        assert_eq!(polynomials[i].num_vars, v, "p_0 and p{} do not have the same number of variables", i+1);
    }

    //println!("Original d = {d}");

    // 1. Base case: if d = 1, g(x) = p_1(x)
    // Map the initial Boolean hypercube {0, 1}^v to the U_1^v grid layout [inf, 0]
    if d == 1 {
        let v = polynomials[0].num_vars();
        let mut current_grid = polynomials[0].evaluations.clone();

        // Apply the bijection dimension by dimension, from lowest stride (X0) to highest
        // To transform an axis from [0, 1] format to the [inf, 0] protocol format:
        // - Value at 0 (new index 1) = old value at 0
        // - Value at inf (new index 0) = derivative slope = (old value at 1) - (old value at 0)
        for var in 0..v {
            let stride = usize::pow(2, var as u32); // Current variable stride size (2^var)
            let chunk_size = stride * 2;
            let mut next_grid = vec![Fr::from(0u64); current_grid.len()];

            for chunk in 0..(current_grid.len() / chunk_size) {
                let offset = chunk * chunk_size;
                for i in 0..stride {
                    let p0 = current_grid[offset + i];          // Evaluation at point 0
                    let p1 = current_grid[offset + stride + i]; // Evaluation at point 1

                    let p_inf = p1 - p0; // Projective limit at infinity (leading coefficient)

                    // Reorder elements into the target [inf, 0] layout structure
                    next_grid[offset + i] = p_inf;          // Axis index 0 (inf position)
                    next_grid[offset + stride + i] = p0;    // Axis index 1 (0 position)
                }
            }
            current_grid = next_grid;
        }
        return current_grid;
    }

    // 2. Divide: split the polynomials into two halves
    let m = d/2;

    //println!("m = {m}");
    
    // Recursive calls for left and right sub-products
    let q_l = multi_product_eval(&polynomials[0..m], m);
    let q_r = multi_product_eval(&polynomials[m..d], d - m);

    /* 
    println!("\nq_l for m = {m}");
    for elmt in &q_l {
        println!("{:?}",elmt);
    }

    println!("\nq_r for m = {m}");
    for elmt in &q_r {
        println!("{:?}",elmt);
    }
    */

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

    /* 
    println!("\nq_l_prime for m = {m}");
    for elmt in &q_l_prime {
        println!("{:?}",elmt);
    }

    println!("\nq_r_prime for m = {m}");
    for elmt in &q_r_prime {
        println!("{:?}",elmt);
    }    
    */

    // 4. Combine: Pointwise product of evaluations (Hadamard product)
    let mut g = Vec::with_capacity(q_l_prime.len());
    for i in 0..q_l_prime.len() {
        g.push(q_l_prime[i] * q_r_prime[i]);
    }

    //println!("Debugging of multi_product_eval : end");

    g
}