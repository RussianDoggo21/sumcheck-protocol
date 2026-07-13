use ark_ff::{BigInteger256, BigInteger384, Field, MontConfig, PrimeField};
use ark_test_curves::bls12_381::{Fr, FrConfig};

use std::sync::atomic::{AtomicU64, Ordering};

pub static FAST_PATH_COUNT: AtomicU64 = AtomicU64::new(0);
pub static SLOW_PATH_COUNT: AtomicU64 = AtomicU64::new(0);
const DELAYED_REDUCTION_THRESHOLD: usize = 14;

// Precomputed constant equal to 2^256 mod p, used to fold the overflow limbs of a delayed
// reduction accumulator back into the field.
lazy_static::lazy_static! {
    static ref R_256: Fr = {
        let r2_bigint = <FrConfig as MontConfig<4>>::R2;
        Fr::new_unchecked(r2_bigint)
    };
}

pub fn extrapolate_dot_product(
    accumulator: &mut Fr,
    window_evals: &[Fr],
    coeff_limbs: &[BigInteger256],
    coefficients: &[Fr],
) {
    if coefficients.len() < DELAYED_REDUCTION_THRESHOLD {
        for i in 0..coefficients.len() {
            *accumulator += window_evals[i] * coefficients[i];
        }
    } else {
        adaptive_dot_product_accumulate_precomputed(accumulator, window_evals, coeff_limbs, coefficients);
    }
}

/// Raw multiply-accumulate step: adds `small * coeff_limbs` (as a plain, unreduced integer
/// product) into the wide accumulator `global_t`, propagating the carry into the overflow limbs.
///
/// This is the arithmetic core shared by every small-big term of a delayed-reduction dot
/// product. It deliberately does NOT construct an `Fr` and does NOT reduce mod p: doing so per
/// term would reintroduce a full Montgomery round-trip on every call, exactly the overhead the
/// delayed-reduction technique exists to avoid.
///
/// `pub(crate)` because benchmark.rs also uses it directly (Sanity Check 1) to measure its
/// standalone, single-term cost against `small_big_mul_raw`. // NEW ! TO UNDERSTAND
#[inline(always)]
pub(crate) fn small_big_mac_raw(global_t: &mut BigInteger384, small: u64, coeff_limbs: &BigInteger256) {
    let small_u128 = small as u128;
    let mut carry: u128 = 0;

    for j in 0..4 {
        let prod = (coeff_limbs.0[j] as u128) * small_u128 + carry + (global_t.0[j] as u128);
        global_t.0[j] = prod as u64;
        carry = prod >> 64;
    }

    let mut j = 4;
    while carry > 0 && j < 6 {
        let sum = (global_t.0[j] as u128) + carry;
        global_t.0[j] = sum as u64;
        carry = sum >> 64;
        j += 1;
    }
}

/// Turns a delayed-reduction accumulator (a plain, unreduced 384-bit integer) back into a
/// proper field element: the low 256 bits are handed to `Fr::new`, which performs the actual
/// mod-p reduction; any overflow into limbs 4-5 is folded back in via the `R_256` correction
/// constant.
///
/// Factored out of `adaptive_dot_product_accumulate_precomputed` so benchmark.rs can reuse the
/// exact same finalization when timing `small_big_mac_raw` on its own. // NEW ! TO UNDERSTAND
#[inline(always)]
pub(crate) fn finalize_delayed_reduction(global_t: &BigInteger384) -> Fr {
    let mut low_limbs = [0u64; 4];
    low_limbs.copy_from_slice(&global_t.0[0..4]);
    let mut result = Fr::new(BigInteger256::new(low_limbs));

    if global_t.0[4] > 0 || global_t.0[5] > 0 {
        let overflow_limbs = BigInteger256::new([global_t.0[4], global_t.0[5], 0, 0]);
        let overflow_fr = Fr::from_bigint(overflow_limbs).unwrap();
        result += overflow_fr * *R_256;
    }

    result
}

/// Computes a fast dot product between a window of Fr elements (which might contain small integers)
/// and precomputed Fr coefficients, accumulating the result into `accumulator`.
///
/// Dynamically routes each term to `small_big_mac_raw` when the element is small, and falls
/// back to full field multiplication otherwise.
pub fn adaptive_dot_product_accumulate_precomputed(
    accumulator: &mut Fr,
    window_evals: &[Fr],
    coeff_limbs: &[BigInteger256],
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
        let bigint = window_evals[i].into_bigint();

        if bigint.0[1] == 0 && bigint.0[2] == 0 && bigint.0[3] == 0 {
            FAST_PATH_COUNT.fetch_add(1, Ordering::Relaxed);
            let small = bigint.0[0];
            if small == 0 {
                continue;
            }
            small_big_mac_raw(&mut global_t, small, &coeff_limbs[i]);
        } else {
            SLOW_PATH_COUNT.fetch_add(1, Ordering::Relaxed);
            slow_path_sum += window_evals[i] * coefficients[i];
        }
    }

    // NEW ! TO UNDERSTAND : finalization now shared with the standalone benchmark via
    // finalize_delayed_reduction, instead of being duplicated inline here.
    *accumulator += finalize_delayed_reduction(&global_t) + slow_path_sum;
}

/// Legacy dot product kept only as the historical reference implementation; no longer used
/// on the hot path (superseded by `adaptive_dot_product_accumulate_precomputed`, which avoids
/// recomputing `coefficients[i].into_bigint()` on every call -- see the earlier profiling
/// session). Kept for documentation purposes / potential future regression tests.
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
        let bigint = window_evals[i].into_bigint();

        if bigint.0[1] == 0 && bigint.0[2] == 0 && bigint.0[3] == 0 {
            FAST_PATH_COUNT.fetch_add(1, Ordering::Relaxed);
            let small = bigint.0[0];
            if small == 0 {
                continue;
            }
            let coeff_limbs = coefficients[i].into_bigint();
            small_big_mac_raw(&mut global_t, small, &coeff_limbs);
        } else {
            SLOW_PATH_COUNT.fetch_add(1, Ordering::Relaxed);
            slow_path_sum += window_evals[i] * coefficients[i];
        }
    }

    *accumulator += finalize_delayed_reduction(&global_t) + slow_path_sum;
}

/// Computes the raw integer product `small * big` (without reduction), then reconstructs a
/// proper field element via delayed Montgomery reconstruction. Standalone, single-call
/// counterpart to `small_big_mac_raw` + `finalize_delayed_reduction` -- only worth using when
/// NOT batched (see Sanity Check 1 in benchmark.rs, which measures exactly this trade-off).
pub fn small_big_mul_raw(small: u64, big: &Fr) -> Fr {
    let big_repr = big.into_bigint();
    let limbs = big_repr.as_ref();

    let mut res_limbs = [0u64; 6];
    let mut carry: u128 = 0;
    let small_u128 = small as u128;

    for i in 0..limbs.len() {
        let product_u128: u128 = (limbs[i] as u128) * small_u128 + carry;
        res_limbs[i] = product_u128 as u64;
        carry = product_u128 >> 64;
    }
    res_limbs[4] = carry as u64;

    let mut low_limbs = [0u64; 4];
    low_limbs.copy_from_slice(&res_limbs[0..4]);
    let mut final_fr = Fr::new(BigInteger256::new(low_limbs));

    if res_limbs[4] > 0 || res_limbs[5] > 0 {
        let overflow_limbs = BigInteger256::new([res_limbs[4], res_limbs[5], 0, 0]);
        let overflow_fr = Fr::from_bigint(overflow_limbs).unwrap_or(Fr::ZERO);
        let r2_bigint = <FrConfig as MontConfig<4>>::R2;
        let r_256_constant = Fr::new_unchecked(r2_bigint);
        final_fr += overflow_fr * r_256_constant;
    }

    final_fr
}