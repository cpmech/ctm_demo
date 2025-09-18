#!/bin/bash

cargo clean && cargo test --features intel_mkl,local_suitesparse
