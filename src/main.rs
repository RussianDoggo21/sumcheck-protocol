// First implementation of sumcheck protocol using arkworks

mod improved;
mod utils;

// NEW ! TO UNDERSTAND : main.rs no longer needs any of the arkworks/std imports that were
// only used by the benchmark logic itself; that logic (and its imports) now lives in benchmark.rs.
use crate::improved::benchmark::{
    run_multiplication_ratio_benchmark, run_all_sc_benchmark, bench_offline_seq_vs_parallel,
    run_all_sc_memory_benchmark,
};

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
    bench_offline_seq_vs_parallel();
    // NEW ! TO UNDERSTAND : same (Degree, Variables) grid again, but measuring peak heap
    // memory instead of wall-clock time, for Arkworks / LinearTimeSC / EvalProductSV.
    run_all_sc_memory_benchmark();
}
