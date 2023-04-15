#!/bin/bash

set -xe

rm -rf zig-cache/*
zig build-lib demo.zig -O ReleaseFast -target x86_64-freestanding -femit-llvm-bc -femit-llvm-ir
