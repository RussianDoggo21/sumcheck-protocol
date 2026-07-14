#!/bin/bash

# NEW ! TO UNDERSTAND : -C target-cpu=native lets rustc emit every instruction the host CPU
# actually supports (in particular BMI2/ADX -- mulx/adcx/adox) instead of the generic
# x86-64 baseline. arkworks' Montgomery field arithmetic (small_big_mac_raw, MontBackend
# mul_assign, etc.) is exactly the kind of 64x64->128 multiply-heavy code that benefits the
# most from this -- typically 20-30% with zero code changes. The resulting binary is tied to
# this machine's CPU (not portable), which is fine for local benchmarking.

RUSTFLAGS="-C target-cpu=native -C force-frame-pointers=yes" 
cargo build --release
sudo perf record --call-graph fp -F 99 -o perf.data -- ./target/release/first_impl
sudo perf report