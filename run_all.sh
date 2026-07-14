#!/bin/bash


# 1. Run the Rust benchmark
echo "=== [1/2] Data extraction with Rust ==="
# NEW ! TO UNDERSTAND : RUSTFLAGS must be set on the SAME command that actually builds/runs
# the binary. Building separately with `cargo build --release` first and then running with a
# plain `cargo run --release` would make cargo detect the changed RUSTFLAGS and rebuild
# WITHOUT target-cpu=native, silently discarding it -- see the -C target-cpu=native note from
# the perf script for why this flag matters here.
RUSTFLAGS="-C target-cpu=native" cargo run --release

# If the benchmark went correctly, we plot the results
if [ $? -eq 0 ]; then
    # NEW ! TO UNDERSTAND : wipe out old results so plot_benchmarks.py only ever
    # picks up fresh data from this run (mkdir -p recreates the folders if needed)
    echo -e "\n=== Cleaning old graphs/ output ==="
    rm -rf graphs
    mkdir -p graphs

    # Generation of the graphs
    echo -e "\n=== [2/2] Graph generation with Python ==="
    python3 plot_benchmarks.py
else
    echo "Error during the Rust benchmark. Python script cancelled."
fi