use ark_ff::{Field, PrimeField,BigInteger256};
use ark_test_curves::bls12_381::Fr;

use crate::improved::arithmetic::extrapolate_dot_product;

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

/// NEW ! TO UNDERSTAND : bundles the 3 pieces of a `compute_kernel(k)` result that
/// `multivariate_extrapolate` needs, so they can be precomputed once and shared (by reference)
/// across every chunk / recursive call that needs the same `k`, instead of recomputing
/// `compute_kernel(k)` (which does `k` modular inversions) from scratch every time.
#[derive(Clone)]
pub struct KernelData {
    pub kernel_inf: Fr,
    pub kernel_classical: Vec<Fr>,
    pub kernel_classical_bigints: Vec<BigInteger256>,
}

/// Precomputes `KernelData` for every k in `1..=max_k`. `multi_product_eval`'s divide-and-
/// conquer recursion only ever calls `multivariate_extrapolate` with k values bounded by `d`
/// (the top-level degree), and always the SAME small set of k values across every chunk of the
/// offline parallel phase -- so computing them once up front (instead of once per chunk, per
/// recursion level) removes a large amount of redundant `compute_kernel` work (perf showed
/// this as part of the `MontConfig::inverse` / `multivariate_extrapolate` overhead).
pub fn precompute_kernel_cache(max_k: usize) -> Vec<Option<KernelData>> {
    let mut cache = vec![None; max_k + 1];
    for k in 1..=max_k {
        let kernel = compute_kernel(k);
        let kernel_inf = kernel[0];
        let kernel_classical: Vec<Fr> = kernel[1..].to_vec();
        let kernel_classical_bigints: Vec<BigInteger256> =
            kernel_classical.iter().map(|c| c.into_bigint()).collect();
        cache[k] = Some(KernelData { kernel_inf, kernel_classical, kernel_classical_bigints });
    }
    cache
}

/// Performs multivariate polynomial extrapolation from U_k^v to U_d^v.
///
/// - `initial_evals`: Flattened evaluations over U_k^v (size must be (k+1)^num_vars)
/// - `k`: Degree parameter / initial window size
/// - `num_extrap`: Number of extra points to compute per axis (d - k)
/// - `num_vars`: Number of variables (v)
/// - `kernel_cache`: precomputed via `precompute_kernel_cache`, must have an entry for `k`
/// - `local_fast` / `local_slow`: per-chunk local path counters (see the note on
///   `adaptive_dot_product_accumulate_precomputed` in arithmetic.rs) -- threaded through rather
///   than touching the global atomics directly, to avoid cross-thread contention.
pub fn multivariate_extrapolate(
    initial_evals: &[Fr],
    k: usize,
    num_extrap: usize,
    num_vars: usize,
    kernel_cache: &[Option<KernelData>],
    local_fast: &mut u64,
    local_slow: &mut u64,
) -> Vec<Fr> {
    let size_k = k + 1;
    let size_d = k + num_extrap + 1;

    assert_eq!(
        initial_evals.len(),
        size_k.pow(num_vars as u32),
        "Initial evaluations vector size does not match (k+1)^v"
    );

    // NEW ! TO UNDERSTAND : looked up instead of recomputed via compute_kernel(k) -- see
    // `precompute_kernel_cache`.
    let kernel_data = kernel_cache[k]
        .as_ref()
        .expect("kernel_cache must be precomputed for every k this call can be reached with");

    // Global retrieving of the coefficients to avoid repeated deferencement
    let kernel_inf = kernel_data.kernel_inf;
    let kernel_classical = &kernel_data.kernel_classical;
    let kernel_classical_bigints = &kernel_data.kernel_classical_bigints;

    // NEW ! TO UNDERSTAND : double-buffering. The cube's size grows monotonically across the
    // `j` loop -- from size_k^(num_vars-1)*size_d at j=1 up to size_d^num_vars at j=num_vars,
    // its largest value (since size_d >= size_k always) -- so instead of allocating a fresh
    // `next_cube` Vec on every single iteration (perf showed this as `Vec::extend_trusted` /
    // `spec_extend` / `from_iter` churn, ~7% of the profile), we allocate exactly two buffers
    // ONCE, sized to that largest capacity, and ping-pong which one is "current" vs "next" by
    // parity of `j`. Each iteration only reads/writes the valid prefix implied by that
    // iteration's own left_variants/right_variants -- the unused tail of each buffer is never
    // touched, so leaving it as stale data from an earlier iteration is harmless.
    let max_size = size_d.pow(num_vars as u32);
    let mut buf_a = vec![Fr::ZERO; max_size];
    let mut buf_b = vec![Fr::ZERO; max_size];
    buf_a[..initial_evals.len()].copy_from_slice(initial_evals);

    // UNIQUE WORK BUFFER BY AXIS (prevent millions of Vec allocations)
    let mut working_row = vec![Fr::ZERO; size_d];

    for j in 1..=num_vars {
        let left_variants = size_d.pow((j - 1) as u32);
        let right_variants = size_k.pow((num_vars - j) as u32);

        let current_axis_stride = right_variants;
        let next_axis_stride = right_variants;

        let (current_cube, next_cube): (&Vec<Fr>, &mut Vec<Fr>) = if j % 2 == 1 {
            (&buf_a, &mut buf_b)
        } else {
            (&buf_b, &mut buf_a)
        };

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
                    extrapolate_dot_product(&mut next_val, &working_row[start_idx..end_idx], kernel_classical_bigints, kernel_classical, local_fast, local_slow);

                    working_row[size_k + c] = next_val;
                }

                // 3. Direct injection from our static buffer into next_cube
                for idx in 0..size_d {
                    let memory_index = next_line_offset + idx * next_axis_stride;
                    next_cube[memory_index] = working_row[idx];
                }
            }
        }
    }

    // NEW ! TO UNDERSTAND : mirrors the ping-pong parity above (loop starts at j=1 writing
    // into buf_b) -- after `num_vars` iterations the final, full-size (size_d^num_vars) result
    // lives in buf_b if num_vars is odd, buf_a if even. No final copy/truncate needed: both
    // buffers are already exactly `max_size` long and the last iteration fills it completely.
    if num_vars % 2 == 1 { buf_b } else { buf_a }
}

/// Implementation of Procedure 1: MultiProductEval
/// Recursively computes the evaluations of g(x) = \prod_{i=1}^d p_i(x) over U_d^v
///
/// # Arguments
/// * `polynomials` - A slice of vectors where each Vec<Fr> contains the evaluations of a polynomial over {0,1}^v
/// * `d` - The total number of polynomials to multiply
/// * `v` - The number of variables for the polynomials in this current sub-cube
/// * `kernel_cache` - precomputed via `precompute_kernel_cache` (shared read-only across every
///   chunk / recursive call -- see the note on `precompute_kernel_cache`)
/// * `local_fast` / `local_slow` - per-chunk local path counters, accumulated (not atomically)
///   across this whole call tree; the caller (one call per chunk) flushes them into the real
///   global atomics ONCE after this returns -- see the note in arithmetic.rs.
pub fn multi_product_eval(polynomials: &[Vec<Fr>], d: usize, v: usize, kernel_cache: &[Option<KernelData>], local_fast: &mut u64, local_slow: &mut u64) -> Vec<Fr> {
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
        let len = polynomials[0].len();
        let u_1_domain = get_u_domain(1); // Contains [Infinity, Value(0)]

        // NEW ! TO UNDERSTAND : double-buffering. Unlike multivariate_extrapolate, this grid's
        // size is CONSTANT across every iteration (always `len` = 2^v -- we're relayouting in
        // place, not growing), so the fix is simpler: allocate exactly two buffers of size
        // `len` ONCE and ping-pong between them by parity of `var`, instead of reallocating a
        // fresh `next_grid` Vec on every one of the `v` iterations.
        let mut buf_a = polynomials[0].clone();
        let mut buf_b = vec![Fr::ZERO; len];

        for var in 0..v {
            // BLACK-BOXED
            let stride = usize::pow(2, var as u32);
            let chunk_size = stride * 2;

            let (current_grid, next_grid): (&Vec<Fr>, &mut Vec<Fr>) = if var % 2 == 0 {
                (&buf_a, &mut buf_b)
            } else {
                (&buf_b, &mut buf_a)
            };

            for chunk in 0..(len / chunk_size) {
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
        }

        // NEW ! TO UNDERSTAND : after `v` iterations (loop starts at var=0 writing into buf_b),
        // the final data lives in buf_a if v is even, buf_b if v is odd.
        return if v % 2 == 0 { buf_a } else { buf_b };
    }

    // 2. Divide: split the polynomials into two halves
    let m = d / 2;

    // Recursive calls for left and right sub-products
    let q_l = multi_product_eval(&polynomials[0..m], m, v, kernel_cache, local_fast, local_slow);
    let q_r = multi_product_eval(&polynomials[m..d], d - m, v, kernel_cache, local_fast, local_slow);

    // 3. Extrapolate both halves to the target domain U_d^v
    // For q_l: currently on U_m^v, needs to reach U_d^v.
    // Number of classical points to add = d - m
    let num_extrap_l = d - m;
    let q_l_prime = multivariate_extrapolate(&q_l, m, num_extrap_l, v, kernel_cache, local_fast, local_slow);

    // For q_r: currently on U_{d-m}^v, needs to reach U_d^v.
    // Number of classical points to add = m
    let num_extrap_r = m;
    let q_r_prime = multivariate_extrapolate(&q_r, d - m, num_extrap_r, v, kernel_cache, local_fast, local_slow);

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


/// NEW ! TO UNDERSTAND : the Lagrange denominators `prod_{j!=i}(x_i - x_j)` for the fixed
/// finite points {0, 1, ..., d-1} never depend on the round's challenge -- only the numerator
/// does. `dynamic_folding_step` used to recompute (and re-invert) them on every single call,
/// i.e. once per round of the early window, even though they're the same `d` values every
/// time. Precompute them ONCE per protocol run (see `online_phase` / `run_with_grid`) and pass
/// them in instead.
pub fn compute_lagrange_denominator_invs<F: Field>(d: usize) -> Vec<F> {
    let classical_points: Vec<F> = (0..d).map(|v| F::from(v as u64)).collect();
    let mut denom_inv = vec![F::zero(); d];
    for i in 0..d {
        let x_i = classical_points[i];
        let mut denominator = F::one();
        for j in 0..d {
            if i != j {
                denominator *= x_i - classical_points[j];
            }
        }
        denom_inv[i] = denominator.inverse().unwrap_or(F::zero());
    }
    denom_inv
}

/// Performs the infinity-aware dynamic folding step over the evaluation grid
#[inline(never)]
pub fn dynamic_folding_step<F: Field>(
    q: &[F],
    challenge: F,
    d: usize,
    base: usize,
    remaining_vars: usize,
    denom_inv: &[F], // NEW ! TO UNDERSTAND : precomputed once per protocol run, not per round
) -> Vec<F> {
    let next_grid_size = usize::pow(base, remaining_vars as u32);
    let mut next_q = vec![F::zero(); next_grid_size];

    // --- PHASE 1: Pre-compute scalars for the unique round challenge ---

    // Finite evaluation points are {0, 1, ..., d-1} (size d)
    let mut classical_points = vec![F::zero(); d];
    for val in 0..d {
        classical_points[val] = F::from(val as u64);
    }

    // 1.(a) Only the numerator (challenge-dependent) part needs recomputing every round;
    // the denominator inverse comes from `denom_inv`, computed once outside this function.
    let mut finite_lagrange_coeffs = vec![F::zero(); d];
    for i in 0..d {
        let mut numerator = F::one();
        for j in 0..d {
            if i != j {
                let x_j = classical_points[j];
                numerator *= challenge - x_j;
            }
        }
        finite_lagrange_coeffs[i] = numerator * denom_inv[i];
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