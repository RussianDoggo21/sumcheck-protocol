use ark_test_curves::bls12_381::Fr;
use crate::improved::arithmetic::fast_dot_product_strided;

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
    let half = small_evals.len() / 2;
    let coeffs = vec![Fr::from(1u64); half]; 

    // Accès direct par enjambement (stride = 2) : ZERO COPIE !
    let p0_r0 = fast_dot_product_strided(small_evals, &coeffs, 0, 2);
    let p1_r0 = fast_dot_product_strided(small_evals, &coeffs, 1, 2);
    proofs.push((p0_r0, p1_r0));

    // Combine evaluations using the first challenge to prepare Round 1
    let r0 = challenges[0];
    let mut current_fr_evals = Vec::with_capacity(half);
    
    for i in 0..half {
        let p0_fr = Fr::from(small_evals[i * 2]);     // Index pair
        let p1_fr = Fr::from(small_evals[i * 2 + 1]); // Index impair
        
        let next_val = p0_fr + r0 * (p1_fr - p0_fr);
        current_fr_evals.push(next_val);
    }

    // --- ROUNDS 1 to num_vars-1: Standard/Hybrid Proving ---
    let mut current_size = half;
    
    for round in 1..num_vars {
        let next_half = current_size / 2;

        // Élimination des vecteurs p0_fr_vec et p1_fr_vec !
        // On accumule directement en lisant dans current_fr_evals par enjambement
        let mut p0 = Fr::from(0u64);
        let mut p1 = Fr::from(0u64);
        
        for i in 0..next_half {
            p0 += current_fr_evals[i * 2];     // Accumulation directe de l'index pair
            p1 += current_fr_evals[i * 2 + 1]; // Accumulation directe de l'index impair
        }
        proofs.push((p0, p1));

        let r = challenges[round];
        let mut next_fr_evals = Vec::with_capacity(next_half);
        
        // On recalcule le niveau suivant en appliquant l'enjambement en ligne
        for i in 0..next_half {
            let p0_val = current_fr_evals[i * 2];
            let p1_val = current_fr_evals[i * 2 + 1];
            
            let next_val = p0_val + r * (p1_val - p0_val);
            next_fr_evals.push(next_val);
        }

        current_fr_evals = next_fr_evals;
        current_size = next_half;
    }

    (claimed_sum, proofs)
}