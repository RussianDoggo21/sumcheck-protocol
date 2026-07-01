use ark_ff::Field;
use ark_ff::UniformRand;
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use ark_std::rand::Rng;
use ark_test_curves::bls12_381::Fr;

pub struct Verifier {
    pub challenges: Vec<Fr>,
    pub s_i_received: Option<Vec<Fr>>, // Stores the current round evaluations of s_i
    pub d: usize,
}

impl Verifier {
    /// Constructor to initialize the Verifier
    pub fn new(d: usize) -> Self {
        Verifier {
            challenges: Vec::new(),
            s_i_received: None,
            d,
        }
    }

    /// Simulates receiving the s_i message from the prover
    pub fn add_s_i(&mut self, s_i: Vec<Fr>) {
        self.s_i_received = Some(s_i);
    }

    /// Generates a random challenge r_i
    pub fn send_challenge<R: Rng>(&mut self, rng: &mut R) -> Fr {
        let r_i = Fr::rand(rng);
        self.challenges.push(r_i);
        r_i
    }

    /// Computes s_i(0) dynamically using the relation s_i(0) := C_{i-1} - s_i(1) (Step 3)
    pub fn compute_s_i_0(&self, c_minus_1: Fr) -> Fr {
        let s_i = self
            .s_i_received
            .as_ref()
            .expect("No s_i polynomial received yet");
        // In U_d_hat = [inf, 1, 2, ..., d-1], the evaluation s_i(1) is at index 1
        let s_i_1 = s_i[1];
        c_minus_1 - s_i_1
    }

    /// Computes C_i := s_i(r_i) by interpolating the received evaluations at point r_i,
    /// strictly implementing Lemma 2 from the paper using the infinity evaluation.
    pub fn update_c_i(&self, r_i: Fr, s_i_0: Fr) -> Fr {
        let s_i_evals = self
            .s_i_received
            .as_ref()
            .expect("No s_i polynomial received yet");

        // According to Lemma 2, our set of d distinct finite evaluation points is:
        // x_1 = 0, x_2 = 1, ..., x_d = d-1
        // Let's build a map of (x_k, s_i(x_k))
        let mut finite_points = vec![(Fr::ZERO, s_i_0)]; // At 0: s_i_0
        for val in 1..self.d {
            // At 1..d-1: s_i_evals[1..d-1]
            finite_points.push((Fr::from(val as u64), s_i_evals[val]));
        }

        // 1. Compute the classical Lagrange interpolation sum over the d finite points (Right term of Eq 6)
        let mut lagrange_sum = Fr::ZERO;
        for i in 0..finite_points.len() {
            let mut numerator = Fr::ONE;
            let mut denominator = Fr::ONE;
            let x_i = finite_points[i].0;

            for j in 0..finite_points.len() {
                if i != j {
                    let x_j = finite_points[j].0;
                    numerator *= r_i - x_j;
                    denominator *= x_i - x_j;
                }
            }

            let l_i = numerator
                * denominator
                    .inverse()
                    .expect("Unexpected zero denominator in Lagrange");
            lagrange_sum += finite_points[i].1 * l_i;
        }

        // 2. Compute the vanishing polynomial product: \prod_{k=1}^d (r_i - x_k) (Left term of Eq 6)
        let mut vanishing_prod = Fr::ONE;
        for i in 0..finite_points.len() {
            vanishing_prod *= r_i - finite_points[i].0;
        }

        // 3. The leading coefficient 'a' is exactly the evaluation at infinity: s_i(\infty) = s_i_evals[0]
        let leading_coeff = s_i_evals[0];

        // Final application of Lemma 2: s_i(r_i) = a * \prod(r_i - x_k) + \sum s_i(x_k)*L_k(r_i)
        let c_i = (leading_coeff * vanishing_prod) + lagrange_sum;

        c_i
    }

    /// Oracle verification at the very end: g(r_1, ..., r_ell) == C_ell
    pub fn final_check(&self, list_of_poly: &[DenseMultilinearExtension<Fr>], c_l: Fr) -> bool {
        assert_eq!(list_of_poly.len(), self.d);

        // Evaluate every single p_k at the collected challenge point (r_1, ..., r_ell)
        let mut g_eval = Fr::ONE;
        for poly in list_of_poly {
            let p_k_eval = poly.evaluate(&self.challenges).unwrap();
            g_eval *= p_k_eval;
        }

        // Verifier accepts if and only if the final evaluation matches C_l
        g_eval == c_l
    }
}
