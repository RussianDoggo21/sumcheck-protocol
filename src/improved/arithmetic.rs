use ark_test_curves::bls12_381::Fr;
use ark_ff::{PrimeField, BigInteger256};

/// Hybrid multiplication between a small integer u64 (64 bits) and a big field element Fr in Montgomery Form
pub fn small_big_mul(small: u64, big: &Fr) -> Fr {
    // Handling of trivial cases
    if small == 0 { return Fr::from(0u64); }
    if small == 1 { return *big; }

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

    let mut carry: u128 = 0;
    let small_u128 = small as u128;

    // We commpute N native multiplications between each limb and the small integer
    // In our case N = 4
    // Since we multiply 64 bits integer together, we use 128 bits format to avoid overflow
    for limb in limbs.iter_mut() {

        let product: u128 = (*limb as u128) * small_u128 + carry; // Computation of said product
        *limb = product as u64; // Update of the current limb (avoid overflow)
        carry = product >> 64; // Carry to add for the next product if the current product is bigger than 64 bits
    }

    // We compute the Montgomery form of the current limbs 
    let mut result = Fr::from_bigint(big_repr).unwrap();
    // In the rare case that the carry of the N-th product isn't equal to 0, we add it to the precedent result
    if carry > 0 {
        let carry_fr = Fr::from(carry) * Fr::from_bigint(BigInteger256::new([0, 0, 0, 1])).unwrap();
        result += carry_fr;
    }

    result
}

/// Produit scalaire rapide combinant Lazy Reduction et Small-Big Multiplication
pub fn fast_dot_product(small_values: &[u64], coefficients: &[Fr]) -> Fr {
    assert_eq!(small_values.len(), coefficients.len(), "Taille incohérente");

    let mut final_res = Fr::from(0u64);
    let mut accumulator_u128: u128 = 0;

    for (&small, coeff) in small_values.iter().zip(coefficients.iter()) {
        if *coeff == Fr::from(1u64) {
            accumulator_u128 += small as u128;
        } else if *coeff == Fr::from(0u64) {
            continue;
        } else {
            if accumulator_u128 > 0 {
                final_res += Fr::from(accumulator_u128);
                accumulator_u128 = 0;
            }
            final_res += small_big_mul(small, coeff);
        }
    }

    if accumulator_u128 > 0 {
        final_res += Fr::from(accumulator_u128);
    }

    final_res
}