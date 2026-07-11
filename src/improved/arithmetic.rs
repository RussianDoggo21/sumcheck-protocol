use ark_ff::{BigInteger256, BigInteger384, Field, MontConfig, PrimeField};
use ark_test_curves::bls12_381::{Fr, FrConfig};

use std::sync::atomic::{AtomicU64, Ordering};

pub static FAST_PATH_COUNT: AtomicU64 = AtomicU64::new(0);
pub static SLOW_PATH_COUNT: AtomicU64 = AtomicU64::new(0);

// Preliminary work
// Heavy computation of the constant 2^256, once and for all
lazy_static::lazy_static! {
    static ref R_256: Fr = {
        let r2_bigint = <FrConfig as MontConfig<4>>::R2;
        Fr::new_unchecked(r2_bigint)
    };
}

/// Computes a fast dot product between a window of Fr elements (which might contain small integers)
/// and precomputed Fr coefficients, accumulating the result into `accumulator`.
///
/// It dynamically optimizes the execution by using `small_big_mul_raw` when the element
/// is small, and falls back to full field multiplication when it has already been extrapolated.
pub fn adaptive_dot_product_accumulate(
    accumulator: &mut Fr,
    window_evals: &[Fr],
    coefficients: &[Fr],
) {
    assert_eq!(
        window_evals.len(),
        coefficients.len(),
        "Size mismatch between window evaluations and coefficients"
    );

    let mut global_t = BigInteger384::zero();
    let mut slow_path_sum = Fr::ZERO;

    for i in 0..coefficients.len() {
        // Optimisation Levier 1 : On récupère les limbs par référence si possible, 
        // ou via into_bigint() si l'API l'impose, mais on évite les manipulations lourdes.
        let bigint = window_evals[i].into_bigint();

        // Levier 2 & 3 : Fusion de la vérification et de la multiplication/accumulation
        if bigint.0[1] == 0 && bigint.0[2] == 0 && bigint.0[3] == 0 {
            FAST_PATH_COUNT.fetch_add(1, Ordering::Relaxed);
            let small = bigint.0[0];
            if small == 0 {
                continue;
            }

            // Inlining + accumulation directe pour éviter l'allocation de BigInteger384 intermédiaire
            let coeff_limbs = coefficients[i].into_bigint();
            let small_u128 = small as u128;
            let mut carry: u128 = 0;

            // On accumule DIRECTEMENT dans les branches de global_t avec gestion du carry étendu
            for j in 0..4 {
                let prod = (coeff_limbs.0[j] as u128) * small_u128 + carry + (global_t.0[j] as u128);
                global_t.0[j] = prod as u64;
                carry = prod >> 64;
            }
            
            // Propagation du carry sur les limbs de débordement de global_t
            let mut j = 4;
            while carry > 0 && j < 6 {
                let sum = (global_t.0[j] as u128) + carry;
                global_t.0[j] = sum as u64;
                carry = sum >> 64;
                j += 1;
            }
        } else {
            // SLOW-PATH
            SLOW_PATH_COUNT.fetch_add(1, Ordering::Relaxed);
            slow_path_sum += window_evals[i] * coefficients[i];
        }
    }

    // --- Finalisation du Fast-Path ---
    let mut low_limbs = [0u64; 4];
    low_limbs.copy_from_slice(&global_t.0[0..4]);
    let mut fast_path_sum = Fr::new(BigInteger256::new(low_limbs));

    if global_t.0[4] > 0 || global_t.0[5] > 0 {
        let overflow_limbs = BigInteger256::new([global_t.0[4], global_t.0[5], 0, 0]);
        let overflow_fr = Fr::from_bigint(overflow_limbs).unwrap();
        fast_path_sum += overflow_fr * *R_256;
    }

    *accumulator += fast_path_sum + slow_path_sum;
}



/// Computes the raw integer product Ti = small * big (WITHOUT reduction)
/// Returns a field element Fr directly using fast delayed reconstruction.
pub fn small_big_mul_raw(small: u64, big: &Fr) -> Fr {
    use ark_ff::{BigInteger256, PrimeField};

    let mut big_repr = big.into_bigint();
    let limbs = big_repr.as_mut();

    let mut res_limbs = [0u64; 6];
    let mut carry: u128 = 0;
    let small_u128 = small as u128;

    for i in 0..limbs.len() {
        let product_u128: u128 = (limbs[i] as u128) * small_u128 + carry;
        res_limbs[i] = product_u128 as u64;
        carry = product_u128 >> 64;
    }
    res_limbs[4] = carry as u64;

    // 1. Extract the 4 lower limbs (first 256 bits of the unreduced result)
    let mut low_limbs = [0u64; 4];
    low_limbs.copy_from_slice(&res_limbs[0..4]);
    
    // Construct the baseline field element directly without instant Montgomery reduction
    let mut final_fr = Fr::new(BigInteger256::new(low_limbs));

    // 2. Handle the overflow remaining in the 5th and 6th limbs (limbs 4 and 5)
    if res_limbs[4] > 0 || res_limbs[5] > 0 {
        let overflow_limbs = BigInteger256::new([res_limbs[4], res_limbs[5], 0, 0]);
        
        // from_bigint applies full modular reduction if the overflow part exceeds the modulus q
        let overflow_fr = Fr::from_bigint(overflow_limbs).unwrap_or(Fr::ZERO);
        
        // Retrieve the constant Montgomery correction factor R_256 (equivalent to 2^256)
        let r2_bigint = <crate::improved::arithmetic::FrConfig as ark_ff::MontConfig<4>>::R2;
        let r_256_constant = Fr::new_unchecked(r2_bigint);

        // Add the scale-corrected overflow component back to the final result
        final_fr += overflow_fr * r_256_constant;
    }

    final_fr
}

/*
/// Quick dot product combining delayed reduction and small-big multiplication
/// Usage of a montgomery reduction  rather than a Barrett reduction
pub fn fast_dot_product_strided(small_values: &[u64], coefficients: &[Fr], offset : usize, stride : usize) -> Fr {
    let required_length = offset + (coefficients.len() - 1) * stride;
    assert!(
        small_values.len() > required_length,
        "The global evaluations array is too small for the requested stride and coefficients"
    );

    // An accumulator for the global giant integer T (pure integer, up to 6 limbs)
    let mut global_t = BigInteger384::from(0u64);

    // We go throught the Fr elements (vector coefficients) with the stride given in parameter
    // We also take into account an offset
    // Useful for the prover in protcol.rs
    // ex: offset = 0, stride = 2 for p0 (0, 2, 4, 6...)
    // ex: offset = 1, stride = 2 for p1 (1, 3, 5, 7...)
    for (i, coeff) in coefficients.iter().enumerate() {
        let idx = offset + i * stride;
        let small = small_values[idx];
        if small == 0 { continue; }

        let t_i = small_big_mul_raw(small, coeff);
        global_t.add_with_carry(&t_i);
    }

    // Optimized delayed reduction

    //1. 4 first limbs -> direct conversion WITHOUT using Montgomery reduction

    // Extraction of the 4 lowest limbs of global_t
    // global_t : 384 bits - too big for Fr::new()
    // low_limbs : 256 bits - small enough for Fr::new()
    let mut low_limbs = [0u64; 4];
    low_limbs.copy_from_slice(&global_t.0[0..4]);

    // new_unchecked() : no Montgomery reduction
    let mut final_sum = Fr::new_unchecked(BigInteger256::new(low_limbs));

    // 2. Separated handling of the 5th and 6th limb (global_t[4], global_t[5])
    if global_t.0[4] > 0 || global_t.0[5] > 0 {
        // We know that t4 and t5 represent a shift of 2^256 bits (so 4 shifts toward the left)
        // We create a BigInt256 containing only the overflow (shift downward)
        let overflow_limbs = BigInteger256::new([
            global_t.0[4],
            global_t.0[5],
            0,
            0
        ]);

        // Conversion of this overflow in field element Fr (Montgomery reduction)
        let overflow_fr = Fr::from_bigint(overflow_limbs).unwrap();

        // Since we shifted the limbs of 4 positions towards the left, we must shift towards the right also 4times
        // We multiply by the constant R_256 (2^256) once to do so
        final_sum += overflow_fr * *R_256;
    }
    final_sum
}
*/
