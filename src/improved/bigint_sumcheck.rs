// Full linear-time sum-check protocol (vanilla / 1-sb / sb-all), built on the raw
// `StdFr2` field of `bigint_field.rs` instead of arkworks' Montgomery-only `Fr`.
//
// This is the integration Christopher (NAIST tutor) asked for: the same protocol
// structure as `LinearTimeSC` in `protocol.rs`/`prover.rs`/`verifier.rs`, but
// re-derived on a field representation that never leaves standard form -- so that
// [DDB26, S3.1]'s small-big multiplication technique is tested under the exact
// assumption it was designed for.
//
// Entry points (`bigint_linear_time_sc_*`) accept the SAME `PolynomialStream<Fr>`
// trait object used throughout the rest of the project (so they can consume the
// existing `MockStream` unchanged), converting to `StdFr2` only at the boundary.

use crate::improved::bigint_field::StdFr2;
use crate::improved::engine::{get_u_hat_domain, EvaluationPoint};
use crate::improved::streaming::PolynomialStream;
use ark_test_curves::bls12_381::Fr;
use rand::Rng;

// ================================================================================
// Prover
// ================================================================================

pub struct BigIntProver {
    list_of_arrays: Vec<Vec<StdFr2>>,
    d: usize,
}

impl BigIntProver {
    pub fn new(list_of_arrays: Vec<Vec<StdFr2>>) -> Self {
        let d = list_of_arrays.len();
        BigIntProver { list_of_arrays, d }
    }

    /// Vanilla: term = (delta_p * StdFr2::from_u64(v)) + p0, using the same generic
    /// big-big multiplication (mul_bb) for both the "scale by v" step and the running
    /// product accumulation. v -> StdFr2::from_u64(v) is hoisted out of the x_prime/k
    /// loops, matching the compute_s_i fix applied to prover.rs.
    pub fn compute_s_i(&self, next_hypercube_size: usize) -> Vec<StdFr2> {
        let u_d_hat = get_u_hat_domain(self.d);
        let mut s_i = vec![StdFr2::zero(); self.d];
        for (u_idx, &u) in u_d_hat.iter().enumerate() {
            let mut sum = StdFr2::zero();
            let v_field = match u {
                EvaluationPoint::Infinity => None,
                EvaluationPoint::Value(v) => Some(StdFr2::from_u64(v)),
            };
            for xp in 0..next_hypercube_size {
                let mut prod = StdFr2::one();
                let i0 = xp << 1;
                let i1 = i0 | 1;
                for k in 0..self.d {
                    let p0 = self.list_of_arrays[k][i0];
                    let p1 = self.list_of_arrays[k][i1];
                    let delta = p1.sub(&p0);
                    let term = match v_field {
                        None => delta,
                        Some(vf) => delta.mul_bb(&vf).add(&p0),
                    };
                    prod = prod.mul_bb(&term);
                }
                sum = sum.add(&prod);
            }
            s_i[u_idx] = sum;
        }
        s_i
    }

    /// Small-big: the "scale delta_p by the small v" step uses StdFr2::mul_sb(v)
    /// (raw small-big, no Montgomery round-trip) instead of a generic mul_bb.
    pub fn compute_s_i_sb(&self, next_hypercube_size: usize) -> Vec<StdFr2> {
        let u_d_hat = get_u_hat_domain(self.d);
        let mut s_i = vec![StdFr2::zero(); self.d];
        for (u_idx, &u) in u_d_hat.iter().enumerate() {
            let mut sum = StdFr2::zero();
            for xp in 0..next_hypercube_size {
                let mut prod = StdFr2::one();
                let i0 = xp << 1;
                let i1 = i0 | 1;
                for k in 0..self.d {
                    let p0 = self.list_of_arrays[k][i0];
                    let p1 = self.list_of_arrays[k][i1];
                    let delta = p1.sub(&p0);
                    let term = match u {
                        EvaluationPoint::Infinity => delta,
                        EvaluationPoint::Value(v) => delta.mul_sb(v).add(&p0),
                    };
                    prod = prod.mul_bb(&term);
                }
                sum = sum.add(&prod);
            }
            s_i[u_idx] = sum;
        }
        s_i
    }

    pub fn update_p_arrays(&mut self, next_hypercube_size: usize, challenge: StdFr2) {
        let mut next_tables = vec![vec![StdFr2::zero(); next_hypercube_size]; self.d];
        for k in 0..self.d {
            for xp in 0..next_hypercube_size {
                let i0 = xp << 1;
                let i1 = i0 | 1;
                let p0 = self.list_of_arrays[k][i0];
                let p1 = self.list_of_arrays[k][i1];
                next_tables[k][xp] = p1.sub(&p0).mul_bb(&challenge).add(&p0);
            }
        }
        self.list_of_arrays = next_tables;
    }
}

// ================================================================================
// Verifier
// ================================================================================

pub struct BigIntVerifier {
    d: usize,
    s_i_evals: Vec<StdFr2>,
    pub challenges: Vec<StdFr2>,
}

impl BigIntVerifier {
    pub fn new(d: usize) -> Self {
        BigIntVerifier { d, s_i_evals: Vec::new(), challenges: Vec::new() }
    }

    pub fn add_s_i(&mut self, s_i: Vec<StdFr2>) {
        self.s_i_evals = s_i;
    }

    /// s_i(0) = c_i - s_i(1), reconstructed since the prover omits both the point at
    /// 0 and (implicitly) needs it recovered from the running claim c_i.
    pub fn compute_s_i_0(&self, c_i: StdFr2) -> StdFr2 {
        let s_i_1 = if self.d >= 2 { self.s_i_evals[1] } else { self.s_i_evals[0] };
        c_i.sub(&s_i_1)
    }

    /// Real, uniformly random challenge (not the deterministic placeholder used in
    /// the earlier standalone prototype).
    pub fn send_challenge<R: Rng + ?Sized>(&mut self, rng: &mut R) -> StdFr2 {
        let r = StdFr2::rand(rng);
        self.challenges.push(r);
        r
    }

    /// Lemma 2 (Lagrange interpolation with the point at infinity): combines s_i(inf)
    /// with the vanishing product, plus the classical Lagrange sum over the d finite
    /// points (0, reconstructed above, and 1..d-1, sent by the prover).
    pub fn update_c_i(&self, challenge: StdFr2, s_i_0: StdFr2) -> StdFr2 {
        let mut classical_points = vec![StdFr2::zero(); self.d];
        let mut classical_values = vec![StdFr2::zero(); self.d];
        classical_points[0] = StdFr2::zero();
        classical_values[0] = s_i_0;
        for v in 1..self.d {
            classical_points[v] = StdFr2::from_u64(v as u64);
            classical_values[v] = self.s_i_evals[v];
        }

        let s_inf = self.s_i_evals[0];

        let mut vanishing = StdFr2::one();
        for &x_k in &classical_points {
            vanishing = vanishing.mul_bb(&challenge.sub(&x_k));
        }

        let mut lagrange_sum = StdFr2::zero();
        for i in 0..self.d {
            let x_i = classical_points[i];
            let mut numerator = StdFr2::one();
            let mut denominator = StdFr2::one();
            for j in 0..self.d {
                if i == j {
                    continue;
                }
                let x_j = classical_points[j];
                numerator = numerator.mul_bb(&challenge.sub(&x_j));
                denominator = denominator.mul_bb(&x_i.sub(&x_j));
            }
            let coeff = numerator.mul_bb(&denominator.inverse());
            lagrange_sum = lagrange_sum.add(&classical_values[i].mul_bb(&coeff));
        }

        s_inf.mul_bb(&vanishing).add(&lagrange_sum)
    }

    /// Final oracle check: evaluate each polynomial's original evaluations at the
    /// accumulated challenge vector via standard multilinear interpolation, then
    /// compare the product to c_i.
    pub fn final_check(&self, original_arrays: &[Vec<StdFr2>], c_i: StdFr2) -> bool {
        let mut product = StdFr2::one();
        for poly in original_arrays {
            let val = evaluate_mle(poly, &self.challenges);
            product = product.mul_bb(&val);
        }
        product == c_i
    }
}

fn evaluate_mle(evals: &[StdFr2], challenges: &[StdFr2]) -> StdFr2 {
    let mut current = evals.to_vec();
    for &r in challenges {
        let half = current.len() / 2;
        let mut next = vec![StdFr2::zero(); half];
        for i in 0..half {
            let p0 = current[2 * i];
            let p1 = current[2 * i + 1];
            next[i] = p1.sub(&p0).mul_bb(&r).add(&p0);
        }
        current = next;
    }
    current[0]
}

// ================================================================================
// Entry points: consume the SAME PolynomialStream<Fr> trait object as LinearTimeSC.
// ================================================================================

fn pull_and_convert(stream: &mut dyn PolynomialStream<Fr>) -> (Vec<Vec<StdFr2>>, usize) {
    let num_vars = stream.num_vars();
    stream.rewind();
    let full_chunk = stream
        .next_chunk(1 << num_vars)
        .expect("Stream should provide the full hypercube data");
    let converted: Vec<Vec<StdFr2>> = full_chunk
        .into_iter()
        .map(|evals| evals.into_iter().map(StdFr2::from).collect())
        .collect();
    (converted, num_vars)
}

/// Vanilla: compute_s_i at every round.
pub fn bigint_linear_time_sc_vanilla(stream: &mut dyn PolynomialStream<Fr>, sumcheck_claim: Fr) -> bool {
    let (list_of_arrays, num_vars) = pull_and_convert(stream);
    let d = list_of_arrays.len();
    let original = list_of_arrays.clone();
    let mut prover = BigIntProver::new(list_of_arrays);
    let mut verifier = BigIntVerifier::new(d);
    let mut rng = rand::thread_rng();
    let mut c_i = StdFr2::from(sumcheck_claim);

    for round in 0..num_vars {
        let next_hyp = (1usize << (num_vars - round)) / 2;
        let s_i = prover.compute_s_i(next_hyp);
        verifier.add_s_i(s_i);
        let s_i_0 = verifier.compute_s_i_0(c_i);
        let challenge = verifier.send_challenge(&mut rng);
        c_i = verifier.update_c_i(challenge, s_i_0);
        prover.update_p_arrays(next_hyp, challenge);
    }

    verifier.final_check(&original, c_i)
}

/// 1-sb: round 0 uses compute_s_i_sb, subsequent rounds use compute_s_i.
/// Mirrors the existing `run_sb_1` / `linear_time_sc_sb_1` naming convention.
pub fn bigint_linear_time_sc_sb1(stream: &mut dyn PolynomialStream<Fr>, sumcheck_claim: Fr) -> bool {
    let (list_of_arrays, num_vars) = pull_and_convert(stream);
    let d = list_of_arrays.len();
    let original = list_of_arrays.clone();
    let mut prover = BigIntProver::new(list_of_arrays);
    let mut verifier = BigIntVerifier::new(d);
    let mut rng = rand::thread_rng();
    let mut c_i = StdFr2::from(sumcheck_claim);

    let next_hyp_0 = (1usize << num_vars) / 2;
    let s_0 = prover.compute_s_i_sb(next_hyp_0);
    verifier.add_s_i(s_0);
    let s_0_0 = verifier.compute_s_i_0(c_i);
    let challenge = verifier.send_challenge(&mut rng);
    c_i = verifier.update_c_i(challenge, s_0_0);
    prover.update_p_arrays(next_hyp_0, challenge);

    for round in 1..num_vars {
        let next_hyp = (1usize << (num_vars - round)) / 2;
        let s_i = prover.compute_s_i(next_hyp);
        verifier.add_s_i(s_i);
        let s_i_0 = verifier.compute_s_i_0(c_i);
        let challenge = verifier.send_challenge(&mut rng);
        c_i = verifier.update_c_i(challenge, s_i_0);
        prover.update_p_arrays(next_hyp, challenge);
    }

    verifier.final_check(&original, c_i)
}

/// sb-all: compute_s_i_sb at EVERY round (extended variant, not in the original
/// naming convention, added to show the cumulative effect across the whole protocol).
pub fn bigint_linear_time_sc_sb_all(stream: &mut dyn PolynomialStream<Fr>, sumcheck_claim: Fr) -> bool {
    let (list_of_arrays, num_vars) = pull_and_convert(stream);
    let d = list_of_arrays.len();
    let original = list_of_arrays.clone();
    let mut prover = BigIntProver::new(list_of_arrays);
    let mut verifier = BigIntVerifier::new(d);
    let mut rng = rand::thread_rng();
    let mut c_i = StdFr2::from(sumcheck_claim);

    for round in 0..num_vars {
        let next_hyp = (1usize << (num_vars - round)) / 2;
        let s_i = prover.compute_s_i_sb(next_hyp);
        verifier.add_s_i(s_i);
        let s_i_0 = verifier.compute_s_i_0(c_i);
        let challenge = verifier.send_challenge(&mut rng);
        c_i = verifier.update_c_i(challenge, s_i_0);
        prover.update_p_arrays(next_hyp, challenge);
    }

    verifier.final_check(&original, c_i)
}