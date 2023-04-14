#!/bin/bash

set -e

cd src/tests/zig
rm -rf zig-cache/*
rm -rf src/zig-cache/*
zig build-lib src/zigtest.zig -O ReleaseFast -target x86_64-freestanding -femit-llvm-bc -femit-llvm-ir
cd -