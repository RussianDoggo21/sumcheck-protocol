use ark_test_curves::bls12_381::Fr;
use ark_ff::{BigInteger, BigInteger384, BigInteger256, PrimeField};

// Preliminary work 
// Heavy computation of constants, once and for all
lazy_static::lazy_static! {
    static ref R_256: Fr = Fr::from_bigint(BigInteger256::new([0, 0, 0, 1])).unwrap(); 
    static ref R_320: Fr = {
        let base_256 = Fr::from_bigint(BigInteger256::new([0, 0, 0, 1])).unwrap();
        let shift_64 = Fr::from(1u64 << 63) * Fr::from(2u64);
        base_256 * shift_64 
    };
}

/// Computes the raw integer product Ti = small * big (WITHOUT reduction)
/// Returns a BigInteger384 (6 limbs) because 4 limbs * 1 limb can grow up to 6 limbs
pub fn small_big_mul_raw(small: u64, big: &Fr) -> BigInteger384 {
    // Originally big is in the Montgomery Form 
    // e.g. big = big' * R mod q, with big' being the actual field element value 
    // and R = 2^256 (the Montgomery constant for a 256-bit field).

    // big_repr returns an object BigInt which represents the raw internal limbs 
    // of the Montgomery representation, encoded as a 256-bit integer.
    // e.g. big.into_bigint() = BigInt([12, 3, 5, 7])
    let mut big_repr = big.into_bigint();

    // as_mut extracts a mutable reference to the underlying u64 array of big_repr
    // e.g. big_repr.as_mut() = &mut [12, 3, 5, 7]
    let limbs = big_repr.as_mut();
    
    let mut res_limbs = [0u64; 6]; // Definition of the limbs to be store the result of each multiplication
    let mut carry: u128 = 0; // Carry of each multiplication
    let small_u128 = small as u128; // Conversion for the multiplication between 2 u128

    // We commpute N native multiplications between each limb and the small integer
    // In our case N = 4
    for i in 0..limbs.len() {
        let product_u128: u128 = (limbs[i] as u128) * small_u128 + carry; // Computation of said product (avoid overflow by storing the result on 128 bits)
        res_limbs[i] = product_u128 as u64; // Storage of the product (first 64 bits of product_u128)
        carry = product_u128 >> 64; // Carry to add for the next product (last 64 bits of product_u128)
    }
    // Store the remaining carry in the 5th limb
    res_limbs[4] = carry as u64;

    BigInteger384::new(res_limbs)
}

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
    // ex: offset = 0, stride = 2 pour p0 (0, 2, 4, 6...)
    // ex: offset = 1, stride = 2 pour p1 (1, 3, 5, 7...)
    for (i, coeff) in coefficients.iter().enumerate() {
        let idx = offset + i * stride;
        let small = small_values[idx];
        if small == 0 { continue; }
        
        let t_i = small_big_mul_raw(small, coeff);
        global_t.add_with_carry(&t_i);
    }

    // Optimized delayed reduction
    let mut final_sum : Fr;

    //1. 4 first limbs -> direct conversion, using arkworks-native Montgomery reduction
    let mut low_limbs = [0u64; 4];
    low_limbs.copy_from_slice(&global_t.0[0..4]);
    final_sum = Fr::from_bigint(BigInteger256::new(low_limbs)).unwrap();

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