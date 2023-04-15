#!/bin/bash

set -xe

export RUST_LOG="k4s=trace"
../../target/release/k4s build --output demo.k4s bootstrap.k4sm demo.bc
