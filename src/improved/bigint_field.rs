// Minimal, from-scratch finite field arithmetic over the BLS12-381 scalar field,
// built DIRECTLY on plain [u64; 4] arrays -- no ark_ff::BigInteger256, no Montgomery
// representation anywhere in this file. Every operation (add, sub, comparison,
// big-big multiplication, small-big multiplication, inversion) is written by hand.
//
// This exists to test [DDB26, S3.1]'s small-big multiplication technique under the
// representational assumption it was actually designed for (standard-form operands
// throughout), after profiling established that arkworks' Montgomery-only `Fr`
// cannot realize the technique's claimed speedup (see the internship report,
// Section "Discussion on Limits, Trade-offs, and Scalability").
//
// `ark_ff`/`ark_test_curves::Fr` are used ONLY at the boundary (converting test/setup
// data in and reading results back out for cross-checking against the trusted
// arkworks implementation) -- never inside any StdFr2 method itself.

use std::cmp::Ordering;

pub type Limbs = [u64; 4];

const MODULUS: Limbs = [
    18446744069414584321,
    6034159408538082302,
    3691218898639771653,
    8353516859464449352,
];

// mu = floor(2^512 / p), 5 nonzero limbs (little-endian), padded to 6 for internal use.
const BARRETT_MU: [u64; 6] = [
    4788304978035696531,
    7279011843745230193,
    4086414915577876179,
    3841734232051169148,
    2,
    0,
];

// ================================================================================
// Raw primitives on [u64; 4] -- nothing here comes from ark_ff.
// ================================================================================

#[inline(always)]
fn raw_cmp(a: &Limbs, b: &Limbs) -> Ordering {
    // Numerically correct comparison: start from the MOST significant limb (index 3).
    // A derived/default array comparison would start from index 0 (least significant
    // in our little-endian layout), which would be WRONG here.
    for i in (0..4).rev() {
        match a[i].cmp(&b[i]) {
            Ordering::Equal => continue,
            other => return other,
        }
    }
    Ordering::Equal
}

#[inline(always)]
fn raw_add(a: &Limbs, b: &Limbs) -> (Limbs, bool) {
    let mut r = [0u64; 4];
    let mut carry: u128 = 0;
    for i in 0..4 {
        let sum = (a[i] as u128) + (b[i] as u128) + carry;
        r[i] = sum as u64;
        carry = sum >> 64;
    }
    (r, carry != 0)
}

#[inline(always)]
fn raw_sub(a: &Limbs, b: &Limbs) -> (Limbs, bool) {
    let mut r = [0u64; 4];
    let mut borrow: i128 = 0;
    for i in 0..4 {
        let diff = (a[i] as i128) - (b[i] as i128) - borrow;
        if diff < 0 {
            r[i] = (diff + (1i128 << 64)) as u64;
            borrow = 1;
        } else {
            r[i] = diff as u64;
            borrow = 0;
        }
    }
    (r, borrow != 0)
}

#[inline(always)]
fn raw_mod_add(a: &Limbs, b: &Limbs) -> Limbs {
    let (mut r, carry) = raw_add(a, b);
    if carry || raw_cmp(&r, &MODULUS) != Ordering::Less {
        let (r2, _) = raw_sub(&r, &MODULUS);
        r = r2;
    }
    r
}

#[inline(always)]
fn raw_mod_sub(a: &Limbs, b: &Limbs) -> Limbs {
    if raw_cmp(a, b) == Ordering::Less {
        let (a_plus_m, _) = raw_add(a, &MODULUS);
        let (r, _) = raw_sub(&a_plus_m, b);
        r
    } else {
        let (r, _) = raw_sub(a, b);
        r
    }
}

/// N x N -> 2N limb multiplication, schoolbook, written by hand.
#[inline(always)]
fn raw_mul_wide(a: &Limbs, b: &Limbs) -> [u64; 8] {
    let mut r = [0u64; 8];
    for i in 0..4 {
        let mut carry: u128 = 0;
        for j in 0..4 {
            let prod = (a[i] as u128) * (b[j] as u128) + (r[i + j] as u128) + carry;
            r[i + j] = prod as u64;
            carry = prod >> 64;
        }
        r[i + 4] = carry as u64;
    }
    r
}

/// Small-big multiplication: N limbs x 1 word -> (N+1) limbs, written by hand.
#[inline(always)]
fn raw_mul_small(a: &Limbs, small: u64) -> [u64; 5] {
    let mut r = [0u64; 5];
    let mut carry: u128 = 0;
    for i in 0..4 {
        let prod = (a[i] as u128) * (small as u128) + carry;
        r[i] = prod as u64;
        carry = prod >> 64;
    }
    r[4] = carry as u64;
    r
}

/// Barrett reduction of a wide (8-limb) product modulo p. HAC Algorithm 14.42.
fn raw_barrett_reduce(x: &[u64; 8]) -> Limbs {
    const K: usize = 4;

    let mut q1 = [0u64; 5];
    q1.copy_from_slice(&x[(K - 1)..8]);

    let mut q2 = [0u64; 11];
    for i in 0..5 {
        let mut carry: u128 = 0;
        for j in 0..5 {
            let prod = (q1[i] as u128) * (BARRETT_MU[j] as u128) + (q2[i + j] as u128) + carry;
            q2[i + j] = prod as u64;
            carry = prod >> 64;
        }
        q2[i + 5] = q2[i + 5].wrapping_add(carry as u64);
    }

    let mut q3 = [0u64; 6];
    q3.copy_from_slice(&q2[5..11]);

    let mut r1 = [0u64; 5];
    r1.copy_from_slice(&x[0..5]);

    let mut r2 = [0u64; 5];
    for i in 0..5 {
        if q3[i] == 0 {
            continue;
        }
        let mut carry: u128 = 0;
        for j in 0..4 {
            if i + j >= 5 {
                break;
            }
            let prod = (q3[i] as u128) * (MODULUS[j] as u128) + (r2[i + j] as u128) + carry;
            r2[i + j] = prod as u64;
            carry = prod >> 64;
        }
        if i + 4 < 5 {
            r2[i + 4] = r2[i + 4].wrapping_add(carry as u64);
        }
    }

    let mut borrow: i128 = 0;
    let mut r = [0u64; 5];
    for i in 0..5 {
        let diff = (r1[i] as i128) - (r2[i] as i128) - borrow;
        if diff < 0 {
            r[i] = (diff + (1i128 << 64)) as u64;
            borrow = 1;
        } else {
            r[i] = diff as u64;
            borrow = 0;
        }
    }

    let mut result: Limbs = [r[0], r[1], r[2], r[3]];
    let mut high = r[4];
    loop {
        if high == 0 {
            break;
        }
        if raw_cmp(&result, &MODULUS) != Ordering::Less {
            let (r2, _) = raw_sub(&result, &MODULUS);
            result = r2;
        } else {
            let mut wide = [result[0], result[1], result[2], result[3], high];
            let mut borrow2: i128 = 0;
            for i in 0..4 {
                let diff = (wide[i] as i128) - (MODULUS[i] as i128) - borrow2;
                if diff < 0 {
                    wide[i] = (diff + (1i128 << 64)) as u64;
                    borrow2 = 1;
                } else {
                    wide[i] = diff as u64;
                    borrow2 = 0;
                }
            }
            wide[4] = wide[4].wrapping_sub(borrow2 as u64);
            result = [wide[0], wide[1], wide[2], wide[3]];
            high = wide[4];
        }
    }
    while raw_cmp(&result, &MODULUS) != Ordering::Less {
        let (r2, _) = raw_sub(&result, &MODULUS);
        result = r2;
    }
    result
}

/// Shared tail of Barrett reduction: given the full 11-limb q2 = q1*mu (or a
/// validly-truncated version thereof, see the two variants below), computes q3, r1,
/// r2 and the final correction loop. Factored out so the two experimental variants
/// below only need to differ in HOW q2 is computed, not in the rest of the algorithm.
fn finish_barrett(x: &[u64; 8], q2: &[u64; 11]) -> Limbs {
    let mut q3 = [0u64; 6];
    q3.copy_from_slice(&q2[5..11]);

    let mut r1 = [0u64; 5];
    r1.copy_from_slice(&x[0..5]);

    let mut r2 = [0u64; 5];
    for i in 0..5 {
        if q3[i] == 0 {
            continue;
        }
        let mut carry: u128 = 0;
        for j in 0..4 {
            if i + j >= 5 {
                break;
            }
            let prod = (q3[i] as u128) * (MODULUS[j] as u128) + (r2[i + j] as u128) + carry;
            r2[i + j] = prod as u64;
            carry = prod >> 64;
        }
        if i + 4 < 5 {
            r2[i + 4] = r2[i + 4].wrapping_add(carry as u64);
        }
    }

    let mut borrow: i128 = 0;
    let mut r = [0u64; 5];
    for i in 0..5 {
        let diff = (r1[i] as i128) - (r2[i] as i128) - borrow;
        if diff < 0 {
            r[i] = (diff + (1i128 << 64)) as u64;
            borrow = 1;
        } else {
            r[i] = diff as u64;
            borrow = 0;
        }
    }

    let mut result: Limbs = [r[0], r[1], r[2], r[3]];
    let mut high = r[4];
    loop {
        if high == 0 {
            break;
        }
        if raw_cmp(&result, &MODULUS) != Ordering::Less {
            let (r2, _) = raw_sub(&result, &MODULUS);
            result = r2;
        } else {
            let mut wide = [result[0], result[1], result[2], result[3], high];
            let mut borrow2: i128 = 0;
            for i in 0..4 {
                let diff = (wide[i] as i128) - (MODULUS[i] as i128) - borrow2;
                if diff < 0 {
                    wide[i] = (diff + (1i128 << 64)) as u64;
                    borrow2 = 1;
                } else {
                    wide[i] = diff as u64;
                    borrow2 = 0;
                }
            }
            wide[4] = wide[4].wrapping_sub(borrow2 as u64);
            result = [wide[0], wide[1], wide[2], wide[3]];
            high = wide[4];
        }
    }
    while raw_cmp(&result, &MODULUS) != Ordering::Less {
        let (r2, _) = raw_sub(&result, &MODULUS);
        result = r2;
    }
    result
}

/// EXPERIMENTAL (see StdFr2::mul_bb_truncated): only computes the (i,j) pairs of
/// q1*mu with i+j>=4 (one limb of slack below the target position 5) -- skip_below=5
/// was verified to break the correction loop's convergence, so 4 is the maximum safe
/// margin. Loop bounds are fully unrolled/static (not a computed j_start) specifically
/// so the compiler has the same opportunity to optimize as the original.
fn raw_barrett_reduce_truncated(x: &[u64; 8]) -> Limbs {
    let mut q1 = [0u64; 5];
    q1.copy_from_slice(&x[3..8]);
    let mut q2 = [0u64; 11];

    // ligne i=0 : seul j=4 contribue (i+j>=4)
    {
        let p = (q1[0] as u128) * (BARRETT_MU[4] as u128) + (q2[4] as u128);
        q2[4] = p as u64;
        q2[5] = q2[5].wrapping_add((p >> 64) as u64);
    }
    // ligne i=1 : j=3,4
    {
        let mut carry: u128 = 0;
        let p = (q1[1] as u128) * (BARRETT_MU[3] as u128) + (q2[4] as u128); q2[4] = p as u64; carry = p >> 64;
        let p = (q1[1] as u128) * (BARRETT_MU[4] as u128) + (q2[5] as u128) + carry; q2[5] = p as u64; carry = p >> 64;
        q2[6] = q2[6].wrapping_add(carry as u64);
    }
    // ligne i=2 : j=2,3,4
    {
        let mut carry: u128 = 0;
        let p = (q1[2] as u128) * (BARRETT_MU[2] as u128) + (q2[4] as u128); q2[4] = p as u64; carry = p >> 64;
        let p = (q1[2] as u128) * (BARRETT_MU[3] as u128) + (q2[5] as u128) + carry; q2[5] = p as u64; carry = p >> 64;
        let p = (q1[2] as u128) * (BARRETT_MU[4] as u128) + (q2[6] as u128) + carry; q2[6] = p as u64; carry = p >> 64;
        q2[7] = q2[7].wrapping_add(carry as u64);
    }
    // ligne i=3 : j=1,2,3,4
    {
        let mut carry: u128 = 0;
        let p = (q1[3] as u128) * (BARRETT_MU[1] as u128) + (q2[4] as u128); q2[4] = p as u64; carry = p >> 64;
        let p = (q1[3] as u128) * (BARRETT_MU[2] as u128) + (q2[5] as u128) + carry; q2[5] = p as u64; carry = p >> 64;
        let p = (q1[3] as u128) * (BARRETT_MU[3] as u128) + (q2[6] as u128) + carry; q2[6] = p as u64; carry = p >> 64;
        let p = (q1[3] as u128) * (BARRETT_MU[4] as u128) + (q2[7] as u128) + carry; q2[7] = p as u64; carry = p >> 64;
        q2[8] = q2[8].wrapping_add(carry as u64);
    }
    // ligne i=4 : j=0,1,2,3,4 (rien a sauter, ligne complete)
    {
        let mut carry: u128 = 0;
        for j in 0..5 {
            let p = (q1[4] as u128) * (BARRETT_MU[j] as u128) + (q2[4 + j] as u128) + carry;
            q2[4 + j] = p as u64;
            carry = p >> 64;
        }
        q2[9] = q2[9].wrapping_add(carry as u64);
    }

    finish_barrett(x, &q2)
}

/// EXPERIMENTAL (see StdFr2::mul_bb_mu4shift): replaces the BARRETT_MU[4]=2 column
/// (a full 128-bit multiply) with a cheap doubling, keeping the outer 5x5 loop shape
/// otherwise identical/regular to raw_barrett_reduce.
fn raw_barrett_reduce_mu4_shift(x: &[u64; 8]) -> Limbs {
    let mut q1 = [0u64; 5];
    q1.copy_from_slice(&x[3..8]);
    let mut q2 = [0u64; 11];
    for i in 0..5 {
        let mut carry: u128 = 0;
        for j in 0..4 {
            let prod = (q1[i] as u128) * (BARRETT_MU[j] as u128) + (q2[i + j] as u128) + carry;
            q2[i + j] = prod as u64;
            carry = prod >> 64;
        }
        // j=4 : BARRETT_MU[4] == 2 exactement -- doublement au lieu d'un multiply 128 bits
        let p = ((q1[i] as u128) << 1) + (q2[i + 4] as u128) + carry;
        q2[i + 4] = p as u64;
        carry = p >> 64;
        q2[i + 5] = q2[i + 5].wrapping_add(carry as u64);
    }
    finish_barrett(x, &q2)
}

// ================================================================================
// StdFr2 : a field element built directly on the raw primitives above.
// ================================================================================

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct StdFr2(pub Limbs);

impl StdFr2 {
    pub fn zero() -> Self {
        StdFr2([0, 0, 0, 0])
    }

    pub fn one() -> Self {
        StdFr2([1, 0, 0, 0])
    }

    pub fn from_u64(x: u64) -> Self {
        StdFr2([x, 0, 0, 0])
    }

    pub fn is_zero(&self) -> bool {
        self.0 == [0, 0, 0, 0]
    }

    pub fn add(&self, other: &Self) -> Self {
        StdFr2(raw_mod_add(&self.0, &other.0))
    }

    pub fn sub(&self, other: &Self) -> Self {
        StdFr2(raw_mod_sub(&self.0, &other.0))
    }

    /// Generic big-big multiplication, directly on the raw limbs.
    pub fn mul_bb(&self, other: &Self) -> Self {
        let wide = raw_mul_wide(&self.0, &other.0);
        StdFr2(raw_barrett_reduce(&wide))
    }

    /// EXPERIMENTAL VARIANT (kept for benchmarking only -- see the report, Section
    /// "Attempted Algorithmic Improvements to raw_barrett_reduce"). Truncates the
    /// q1*mu product in Barrett reduction, skipping the (i,j) pairs with i+j<4 that
    /// never contribute to the retained high limbs -- theoretically ~40% fewer
    /// multiplications for that step (15 instead of 25), verified correct on 800,000+
    /// random pairs plus edge cases, and skip_below=5 was verified to break the
    /// correction loop's convergence (the maximum safe margin, matching HAC's slack).
    /// Despite being correct and doing genuinely less arithmetic, it measured slower
    /// than mul_bb once averaged over multiple runs: the fully regular, compile-time
    /// -constant-bounds loop of raw_barrett_reduce is apparently already very well
    /// optimized by LLVM, and this hand-unrolled, irregular-row-length replacement
    /// loses more from disrupting that regularity than it gains from fewer operations.
    pub fn mul_bb_truncated(&self, other: &Self) -> Self {
        let wide = raw_mul_wide(&self.0, &other.0);
        StdFr2(raw_barrett_reduce_truncated(&wide))
    }

    /// EXPERIMENTAL VARIANT (kept for benchmarking only -- see the report). Replaces
    /// the BARRETT_MU[4]=2 column of the q1*mu computation (a full 128-bit multiply)
    /// with a cheap bit-shift (doubling), since mu's top limb happens to be exactly 2
    /// for the BLS12-381 scalar field modulus -- keeping the rest of the loop's shape
    /// fully regular/unchanged, unlike mul_bb_truncated above. Verified correct on
    /// 300,000+ random pairs. Also measured slower than mul_bb once averaged over
    /// multiple runs, for the same likely reason (compiler optimization of the
    /// original regular loop already outperforms hand-tuned irregular variants here).
    pub fn mul_bb_mu4shift(&self, other: &Self) -> Self {
        let wide = raw_mul_wide(&self.0, &other.0);
        StdFr2(raw_barrett_reduce_mu4_shift(&wide))
    }

    /// Small-big multiplication, directly on the raw limbs (no Montgomery round-trip).
    pub fn mul_sb(&self, small: u64) -> Self {
        let res = raw_mul_small(&self.0, small);
        let mut wide = [0u64; 8];
        wide[0..5].copy_from_slice(&res);
        StdFr2(raw_barrett_reduce(&wide))
    }

    /// Modular inversion via Fermat's little theorem (a^(p-2) mod p), binary
    /// exponentiation, reusing only mul_bb. Rarely called (a handful of times per
    /// round, not per term), so its performance is not a design concern here.
    pub fn inverse(&self) -> Self {
        let (exponent, borrow) = raw_sub(&MODULUS, &[2, 0, 0, 0]);
        debug_assert!(!borrow);

        let mut result = StdFr2::one();
        let mut base = *self;

        for limb_idx in 0..4 {
            let mut limb = exponent[limb_idx];
            for _bit in 0..64 {
                if limb & 1 == 1 {
                    result = result.mul_bb(&base);
                }
                base = base.mul_bb(&base);
                limb >>= 1;
            }
        }
        result
    }

    /// Uniformly random field element via rejection sampling.
    pub fn rand<R: rand::Rng + ?Sized>(rng: &mut R) -> Self {
        loop {
            // NEW ! TO UNDERSTAND : `gen` is a reserved keyword starting with the 2024
            // edition (reserved for generator blocks), so `Rng::gen` must be called via
            // the raw identifier `r#gen` here -- same method, same behavior, just escaped
            // syntax. (`rng.gen_range(0..=u64::MAX)` would also compile, since `gen_range`
            // is a distinct identifier, not the bare `gen` token -- but it routes through
            // rand's `Uniform` sampler instead of a direct fill, for an equivalent result
            // via a different internal path; no reason to prefer it here.)
            let limbs: Limbs = [rng.r#gen(), rng.r#gen(), rng.r#gen(), rng.r#gen()];
            if raw_cmp(&limbs, &MODULUS) == Ordering::Less {
                return StdFr2(limbs);
            }
        }
    }
}

// ================================================================================
// Conversions to/from arkworks' Fr -- ONLY used at the boundary (test/setup data in,
// results read back out). Never used inside any StdFr2 method above.
// ================================================================================

use ark_ff::PrimeField;
use ark_test_curves::bls12_381::Fr as ArkFr;

impl From<ArkFr> for StdFr2 {
    fn from(f: ArkFr) -> Self {
        StdFr2(f.into_bigint().0)
    }
}

impl StdFr2 {
    pub fn to_ark(&self) -> ArkFr {
        ArkFr::from_bigint(ark_ff::BigInteger256::new(self.0)).expect("value must be canonical (< modulus)")
    }
}