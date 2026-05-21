use ark_test_curves::bls12_381::Fr;
use crate::improved::arithmetic::fast_dot_product;

/// Optimized "Small-Value" Sumcheck protocol implementation for a multilinear polynomial
pub fn sc_protocol_improved(num_vars: usize, small_evals: &[u64]) -> (Fr, Vec<(Fr, Fr)>) {
    let mut proofs = Vec::with_capacity(num_vars);
    
    // Compute the initial total claimed sum using delayed reduction
    let mut total_sum_u128: u128 = 0;
    for &val in small_evals {
        total_sum_u128 += val as u128;
    }
    let claimed_sum = Fr::from(total_sum_u128);

    // Fixed deterministic challenges for verification simulation
    let mut challenges = Vec::with_capacity(num_vars);
    for i in 0..num_vars {
        challenges.push(Fr::from((i + 2) as u64));
    }

    // --- ROUND 0: Pure Small-Value Optimization ---
    // At the very first round, all evaluations are guaranteed to be raw u64
    let half = small_evals.len() / 2;
    let mut p0_small = Vec::with_capacity(half);
    let mut p1_small = Vec::with_capacity(half);

    for i in 0..half {
        p0_small.push(small_evals[i * 2]);
        p1_small.push(small_evals[i * 2 + 1]);
    }

    let coeffs = vec![Fr::from(1u64); half];
    let p0_r0 = fast_dot_product(&p0_small, &coeffs);
    let p1_r0 = fast_dot_product(&p1_small, &coeffs);
    proofs.push((p0_r0, p1_r0));

    // Combine evaluations using the first challenge to prepare Round 1
    // From this point onward, evaluations upgrade to full Fr elements
    let r0 = challenges[0];
    let mut current_fr_evals = Vec::with_capacity(half);
    for i in 0..half {
        let p0_fr = Fr::from(p0_small[i]);
        let p1_fr = Fr::from(p1_small[i]);
        let next_val = p0_fr + r0 * (p1_fr - p0_fr);
        current_fr_evals.push(next_val);
    }

    // --- ROUNDS 1 to num_vars-1: Standard/Hybrid Proving ---
    // Remaining rounds process the upgraded Fr elements sequentially
    let mut current_size = half;
    
    for round in 1..num_vars {
        let next_half = current_size / 2;
        let mut p0_fr_vec = Vec::with_capacity(next_half);
        let mut p1_fr_vec = Vec::with_capacity(next_half);

        for i in 0..next_half {
            p0_fr_vec.push(current_fr_evals[i * 2]);
            p1_fr_vec.push(current_fr_evals[i * 2 + 1]);
        }

        let mut p0 = Fr::from(0u64);
        let mut p1 = Fr::from(0u64);
        for i in 0..next_half {
            p0 += p0_fr_vec[i];
            p1 += p1_fr_vec[i];
        }
        proofs.push((p0, p1));

        let r = challenges[round];
        let mut next_fr_evals = Vec::with_capacity(next_half);
        for i in 0..next_half {
            let next_val = p0_fr_vec[i] + r * (p1_fr_vec[i] - p0_fr_vec[i]);
            next_fr_evals.push(next_val);
        }

        current_fr_evals = next_fr_evals;
        current_size = next_half;
    }

    (claimed_sum, proofs)
}