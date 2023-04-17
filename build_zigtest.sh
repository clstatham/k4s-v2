#!/bin/bash

set -e

cd src/tests/zig
rm -rf zig-cache/*
rm -rf src/zig-cache/*
zig build-lib src/bootloader.zig -O ReleaseSmall -target x86_64-freestanding --strip  -fLLVM -femit-llvm-bc -femit-llvm-ir -fno-lto
zig build-lib src/kernel.zig -O ReleaseSmall -target x86_64-freestanding --strip  -fLLVM -femit-llvm-bc -femit-llvm-ir -fno-lto
cd -
RUST_LOG="k4s=debug" cargo run -- build --output target/kernel.k4s src/tests/zig/bootstrap.k4sm src/tests/zig/bootloader.bc src/tests/zig/kernel.bc