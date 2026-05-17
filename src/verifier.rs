// Contains the logic of the verifier
use ark_test_curves::bls12_381::Fr;
use ark_poly::univariate::DensePolynomial;
use ark_poly::Polynomial;
use ark_std::rand::Rng;

// Verifier algorithm at the i-th round
pub fn verifier_i(g_i: &DensePolynomial<Fr>, current_claim: Fr) -> Result<Fr, &'static str> {
    let eval_0 = g_i.evaluate(&Fr::from(0));
    let eval_1 = g_i.evaluate(&Fr::from(1));
    let check_sum = eval_0 + eval_1;

    // If the check fails, we stop the programm : the Prover is cheating
    if check_sum != current_claim {
        return Err("Sumcheck verification failed: g_i(0) + g_i(1) != current_claim");
    };
    // If the check pass, V "sends" a random field element w_i

    let mut rng = ark_std::test_rng();

    // TEMPORARY CODE
    let random_small_int: u64 = rng.gen_range(0..=100); // TO DELETE
    let w_i = Fr::from(random_small_int); // TO REPLACE WITH Fr::rand(&mut rng)
    // TEMPORARY CODE


    Ok(w_i)
}
