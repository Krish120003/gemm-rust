# GEMM implemented in Rust

Ths is an implementation of GEMM in Rust. The goal is to optimize it to be somewhat competitive with numpy.

## Usage

```bash
$ python3 baseline.py # requires numpy, to generate a baseline for correctness
$ cargo run --release
```

## Current performance

These performance numbers are for a 1024x1024 matrix multiplication on a 10 Core M1 Pro MacBook Pro.

Both Numpy and Rust are limited to a single core to compare performance.

Numpy: 72.3 GFLOP/s \
Rust: 0.95 GFLOP/s

(Long way to go!)
