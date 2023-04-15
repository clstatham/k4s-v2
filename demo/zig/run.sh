#!/bin/bash

set -xe

export RUST_LOG="k4s=debug"
../../target/release/k4s run demo.k4s
