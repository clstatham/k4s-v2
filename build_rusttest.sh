#!/bin/bash

set -e

cd src/tests/rust
cargo clean
RUSTFLAGS="--emit=llvm-bc,llvm-ir" cargo build --release
cd -