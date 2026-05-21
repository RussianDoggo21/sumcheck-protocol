use ark_test_curves::bls12_381::Fr;
use ark_ff::{PrimeField, BigInteger256};

/// Multiplication hybride rapide entre un petit entier et un grand Fr Montgomery
pub fn small_big_mul(small: u64, big: &Fr) -> Fr {
    if small == 0 { return Fr::from(0u64); }
    if small == 1 { return *big; }

    let mut big_repr = big.into_bigint();
    let limbs = big_repr.as_mut();

    let mut carry: u128 = 0;
    let small_u128 = small as u128;

    for limb in limbs.iter_mut() {
        let product: u128 = (*limb as u128) * small_u128 + carry;
        *limb = product as u64;
        carry = product >> 64;
    }

    let mut result = Fr::from_bigint(big_repr).unwrap();

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