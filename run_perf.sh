#!/bin/bash

cargo build --release
sudo perf record --call-graph dwarf ./target/release/first_impl
sudo perf report