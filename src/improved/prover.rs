use crate::improved::engine::{EvaluationPoint, get_u_hat_domain};
use crate::improved::arithmetic::small_big_mul_raw;
use ark_ff::Field;
use ark_poly::DenseMultilinearExtension;
use ark_test_curves::bls12_381::Fr; // Ajuste le chemin selon ton projet

pub struct Prover {
    pub list_of_arrays: Vec<Vec<Fr>>,
    pub d: usize,
}

impl Prover {
    /// Constructor to initialize the Prover state (Step 0)
    pub fn new(list_of_poly: &[DenseMultilinearExtension<Fr>]) -> Self {
        let d = list_of_poly.len();
        // Extract the underlying flat vectors from the multilinear extensions
        let list_of_arrays = list_of_poly
            .iter()
            .map(|poly| poly.evaluations.clone())
            .collect();

        Prover { list_of_arrays, d }
    }

    /// Flexible constructor allowing initialization from pre-folded sub-hypercube arrays.
    /// Essential for transitioning seamlessly into Phase 2 without re-allocating full polynomials.
    pub fn with_arrays(list_of_arrays: Vec<Vec<Fr>>) -> Self {
        let d = list_of_arrays.len();
        Prover { list_of_arrays, d }
    }

    /// Computes the round polynomial s_i(u) over U_d_hat (Step 1)
    pub fn compute_s_i(&self, num_vars: usize, round: usize) -> Vec<Fr> {
        let u_d_hat = get_u_hat_domain(self.d);
        let current_hypercube_size = 1 << (num_vars - round);
        let next_hypercube_size = current_hypercube_size / 2;

        let mut s_i_evals = vec![Fr::ZERO; self.d];

        for (u_idx, &u) in u_d_hat.iter().enumerate() {
            let mut sum_over_hypercube = Fr::ZERO;

            // NEW ! TO UNDERSTAND : v -> Fr::from(v) only depends on u_idx (fixed for this
            // whole u_idx iteration), not on x_prime or k -- hoisted out of the double loop
            // below instead of being recomputed O(next_hypercube_size * d) times per u_idx
            // (each call previously invoked a MontConfig::from_bigint conversion). This is
            // the single largest measured perf win of the whole optimization pass
            // (compute_s_i self-time /3.6, MontConfig::from_bigint disappearing from the
            // profile entirely) -- it had regressed back to the per-term form in this branch.
            let v_as_fr = match u {
                EvaluationPoint::Infinity => None,
                EvaluationPoint::Value(v) => Some(Fr::from(v)),
            };

            for x_prime in 0..next_hypercube_size {
                let mut product_over_k = Fr::ONE;
                let idx_0 = x_prime << 1;
                let idx_1 = idx_0 | 1;

                for k in 0..self.d {
                    let p0 = self.list_of_arrays[k][idx_0];
                    let p1 = self.list_of_arrays[k][idx_1];
                    let delta_p = p1 - p0;

                    let term = match v_as_fr {
                        None => delta_p,
                        Some(v_fr) => (delta_p * v_fr) + p0,
                    };
                    product_over_k *= term;
                }
                sum_over_hypercube += product_over_k;
            }
            s_i_evals[u_idx] = sum_over_hypercube;
        }
        s_i_evals
    }

    pub fn compute_s_0_sb(&self, num_vars: usize) -> Vec<Fr> {
        let u_d_hat = get_u_hat_domain(self.d);
        let current_hypercube_size = 1 << num_vars;
        let next_hypercube_size = current_hypercube_size / 2;

        let mut s_i_evals = vec![Fr::ZERO; self.d];

        for (u_idx, &u) in u_d_hat.iter().enumerate() {
            let mut sum_over_hypercube = Fr::ZERO;

            for x_prime in 0..next_hypercube_size {
                let mut product_over_k = Fr::ONE;
                let idx_0 = x_prime << 1;
                let idx_1 = idx_0 | 1;

                for k in 0..self.d {
                    let p0 = self.list_of_arrays[k][idx_0];
                    let p1 = self.list_of_arrays[k][idx_1];
                    let delta_p = p1 - p0;

                    let term = match u {
                        EvaluationPoint::Infinity => delta_p,
                        EvaluationPoint::Value(v) => small_big_mul_raw(v,&delta_p) + p0,
                    };
                    product_over_k *= term;
                }
                sum_over_hypercube += product_over_k;
            }
            s_i_evals[u_idx] = sum_over_hypercube;
        }
        s_i_evals
    }

    /// Updates the bookkeeping arrays using the verifier's challenge r_i (Step 4)
    pub fn update_p_arrays(&mut self, num_vars: usize, round: usize, challenge: Fr) {
        let current_hypercube_size = 1 << (num_vars - round);
        let next_hypercube_size = current_hypercube_size / 2;

        let mut next_tables = vec![vec![Fr::ZERO; next_hypercube_size]; self.d];
        for k in 0..self.d {
            for x_prime in 0..next_hypercube_size {
                let idx_0 = x_prime << 1;
                let idx_1 = idx_0 | 1;

                let p0 = self.list_of_arrays[k][idx_0];
                let p1 = self.list_of_arrays[k][idx_1];

                next_tables[k][x_prime] = ((p1 - p0) * challenge) + p0;
            }
        }
        self.list_of_arrays = next_tables;
    }
}