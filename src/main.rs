// First implementation of sumcheck protocol using arkworks

mod improved;
mod utils;

// NEW ! TO UNDERSTAND : main.rs no longer needs any of the arkworks/std imports that were
// only used by the benchmark logic itself; that logic (and its imports) now lives in benchmark.rs.
use crate::improved::benchmark::{run_multiplication_ratio_benchmark, run_all_sc_benchmark, bench_offline_seq_vs_parallel};

fn main() {
    // NEW ! TO UNDERSTAND : main() is now just a thin sequence of top-level benchmark calls,
    // all the actual logic lives in benchmark.rs
    run_multiplication_ratio_benchmark();
    run_all_sc_benchmark();
    // NEW ! TO UNDERSTAND : bench_offline_seq_vs_parallel now sweeps internally over the
    // same (Degree, Variables) grid as run_all_sc_benchmark, so it no longer takes args.
    bench_offline_seq_vs_parallel();
}
