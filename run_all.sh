#!/bin/bash

# 1. Run the Rust benchmark
echo "=== [1/2] Data extraction with Rust ==="
cargo run --release

# cargo build --release
# sudo perf record --call-graph dwarf ./target/release/first_impl
# perf report

# If the benchmark went correctly, we plot the results
if [ $? -eq 0 ]; then
    # Generation of the graphs
    echo -e "\n=== [2/2] Graph generation with Python ==="
    python3 plot_benchmarks.py
else
    echo "Error during the Rust benchmark. Python script cancelled."
fi