use ark_poly::DenseMultilinearExtension;
use ark_test_curves::bls12_381::Fr;
use crate::improved::prover::Prover;
use crate::improved::verifier::Verifier;

pub fn linear_time_sc(
    list_of_poly: &[DenseMultilinearExtension<Fr>],
    num_poly: usize,
    sumcheck_claim: Fr,
) -> bool {
    assert!(list_of_poly.len() > 0, "Cannot run sumcheck on an empty list of polynomials");

    let num_vars = list_of_poly[0].num_vars;
    for p in list_of_poly {
        assert_eq!(
            num_vars, p.num_vars,
            "All polynomials must have the same number of variables"
        );
    }

    assert_eq!(
        list_of_poly.len(),
        num_poly,
        "Number of multilinear polynomials must equal {num_poly}"
    );

    // Initialisation of both prover and verifier via their respective constructor
    let mut prover = Prover::new(list_of_poly);
    let mut verifier = Verifier::new(num_poly);

    // C_0 is our initial sumcheck claim
    let mut c_i = sumcheck_claim;

    // Loop through each round i = 0 ... num_vars - 1 (Rounds 1 to l)
    for i in 0..num_vars {
        // 1. Prover computes the s_i evaluations and sends them to Verifier
        let s_i = prover.compute_s_i(num_vars, i);
        verifier.add_s_i(s_i);

        // 2. Verifier computes s_i(0) and samples a random challenge r_i
        let s_i_0 = verifier.compute_s_i_0(c_i);
        let challenge = verifier.send_challenge();

        // 3. Verifier updates their local target claim: C_i = s_i(r_i)
        c_i = verifier.update_c_i(challenge, s_i_0);

        // 4. Prover folds their internal bookkeeping tables down by half using r_i
        prover.update_p_arrays(num_vars, i, challenge);
    }

    // Final oracle evaluation check at the end of the protocol
    verifier.final_check(list_of_poly, c_i)
}