// First implementation of sumcheck protocol using arkworks

mod improved;
mod naive;
mod utils;

// NEW ! TO UNDERSTAND : main.rs no longer needs any of the arkworks/std imports that were
// only used by the benchmark logic itself; that logic (and its imports) now lives in benchmark.rs.
use crate::improved::benchmark::{
    run_multiplication_ratio_benchmark, run_all_sc_benchmark, bench_run_seq_vs_parallel,
    run_all_sc_memory_benchmark, bench_bigint_vanilla_vs_sb, bench_bigint_memory, bench_mul_bb_vs_arkworks, bench_barrett_variants,
};
use crate::naive::benchmark::bench_naive_vs_arkworks_vs_optimized;

// NEW ! TO UNDERSTAND : the memory benchmark needs a way to measure how much heap memory
// each protocol allocates. We install `peak_alloc::PeakAlloc` as the process-wide global
// allocator: it wraps the System allocator and simply tracks currently-allocated bytes and
// the max ("peak") value seen since the last reset. This has to be declared exactly once,
// here in the binary root, so that benchmark.rs (via `crate::PEAK_ALLOC`) can read it around
// whichever call it wants to profile. It replaces the System allocator globally, so it also
// (harmlessly) tracks every other allocation in the program, not just the benchmarked calls.
use peak_alloc::PeakAlloc;

#[global_allocator]
pub static PEAK_ALLOC: PeakAlloc = PeakAlloc;

fn main() {
    // NEW ! TO UNDERSTAND : main() is now just a thin sequence of top-level benchmark calls,
    // all the actual logic lives in benchmark.rs

    run_multiplication_ratio_benchmark();
    run_all_sc_benchmark();
    // NEW ! TO UNDERSTAND : bench_offline_seq_vs_parallel now sweeps internally over the
    // same (Degree, Variables) grid as run_all_sc_benchmark, so it no longer takes args.
    bench_run_seq_vs_parallel();
    // NEW ! TO UNDERSTAND : same (Degree, Variables) grid again, but measuring peak heap
    // memory instead of wall-clock time, for Arkworks / LinearTimeSC / EvalProductSV.
    run_all_sc_memory_benchmark();
    // NEW ! TO UNDERSTAND : full-protocol vanilla vs 1-sb vs sb-all, on the raw BigInt
    // field (bigint_field.rs / bigint_sumcheck.rs) instead of arkworks' Montgomery-only
    // Fr -- see the note above bench_bigint_vanilla_vs_sb in benchmark.rs. Now swept
    // across the same (Degree, Variables) grid as the rest of the benchmarks.
    bench_bigint_vanilla_vs_sb();
    bench_bigint_memory();
    // NEW ! TO UNDERSTAND : isolates exactly how much of the BigInt-vs-arkworks gap in the
    // Section 1/5 curve overlays is due to our Barrett reduction (mul_bb) being less mature
    // than arkworks' assembly-optimized Montgomery multiplication -- see benchmark.rs Section 7.
    bench_mul_bb_vs_arkworks();
    // NEW ! TO UNDERSTAND : two attempted algorithmic optimizations of raw_barrett_reduce
    // (mul_bb_truncated, mul_bb_mu4shift), both correctness-verified but both measured
    // slower than the baseline once averaged -- see bigint_field.rs doc comments and the
    // report.
    bench_barrett_variants();
    // NEW ! TO UNDERSTAND : the d=1 (single multilinear polynomial) sanity check --
    // naive from-scratch vs arkworks vs the current LinearTimeSC, the project's very
    // first working milestone, benchmarked separately since it exercises the naive
    // module's SparsePolynomial-based code path. See naive/benchmark.rs.
    bench_naive_vs_arkworks_vs_optimized();
}