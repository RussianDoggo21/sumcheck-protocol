use ark_poly::{MultilinearExtension, DenseMultilinearExtension};
use ark_ff::PrimeField;

/// A trait representing a stream of evaluations for d polynomials.
/// This allows the Prover to consume data chunk by chunk without loading 
/// the entire hypercube into RAM.
pub trait PolynomialStream<F: PrimeField> {
    /// Returns the total number of variables in the global protocol (l or v).
    fn num_vars(&self) -> usize;

    /// Returns the number of polynomials to multiply (d).
    fn degree(&self) -> usize;

    /// Extracts the next chunk of evaluations of size `chunk_size` (which will be 2^w_t).
    /// Returns a Vector containing d sub-vectors of size `chunk_size`.
    /// Returns `None` when the entire stream has been processed (2^l elements).
    fn next_chunk(&mut self, chunk_size: usize) -> Option<Vec<Vec<F>>>;

    /// Resets the stream cursor back to the beginning.
    /// This is crucial because the algorithm rewinds and re-reads the full stream 
    /// at each new window step 't'.
    fn rewind(&mut self);

    /// Evaluates all underlying polynomials at a complete multi-variate point (r_1, ..., r_l)
    /// without keeping the full hypercube in memory. Required for the final oracle step.
    fn evaluate_at_point(&self, point: &[F]) -> Vec<F>;
}

/// A mock implementation of `PolynomialStream` backed by Arkworks MLEs for unit testing.
/// It avoids duplicating the full data in memory by holding a slice reference.
pub struct MockStream<'a, F: PrimeField> {
    pub l: usize,
    pub d: usize,
    pub data: &'a [DenseMultilinearExtension<F>],
    pub cursor: usize,
}

impl<'a, F: PrimeField> MockStream<'a, F> {
    pub fn new(l: usize, d: usize, data: &'a [DenseMultilinearExtension<F>]) -> Self {
        assert_eq!(data.len(), d, "The number of MLEs must match d");
        for poly in data {
            assert_eq!(poly.num_vars(), l, "All MLEs must have l variables");
        }
        Self { l, d, data, cursor: 0 }
    }
}

impl<'a, F: PrimeField> PolynomialStream<F> for MockStream<'a, F> {
    fn num_vars(&self) -> usize { self.l }
    fn degree(&self) -> usize { self.d }

    fn next_chunk(&mut self, chunk_size: usize) -> Option<Vec<Vec<F>>> {
        let total_size = 1 << self.l; // 2^l := #({0,1}^l)

        // If we have already processed the entire stream 
        // i.e. all the chunks have been extracted
        if self.cursor >= total_size {
            return None;
        }

        let mut chunk_result = Vec::with_capacity(self.d);

        // Size of each chunk : end-start
        // Either equal to chunk_size or less 
        // (if we reach the end of the evaluation vector of a poly)
        let start = self.cursor; // self.cursor will be equal to 0, chunk_size, 2.chunk_size, ..., m.chunk_size
        let end = (start + chunk_size).min(total_size); // end = min(start + chunk_size, total_size)

        for poly in self.data {
            // Directly slice into the internal evaluations vector of the Arkworks MLE
            let chunk_poly = poly.evaluations[start..end].to_vec();
            chunk_result.push(chunk_poly);
        }

        self.cursor += chunk_size;
        Some(chunk_result)
    }

    fn rewind(&mut self) {
        self.cursor = 0;
    }

    fn evaluate_at_point(&self, point: &[F]) -> Vec<F> {
        // Leverages Arkworks internal MLE evaluation for mock testing
        self.data.iter().map(|poly| poly.evaluate(point).unwrap()).collect()
    }
}