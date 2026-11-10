#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/dyq/workspace/reports/baseline-integration-20261026/nn_misc_math
P=$ROOT/private
mkdir -p "$ROOT/results"
run_one(){ local b=$1 o=$2 c=$3; local d="$ROOT/results/${b}__${o}__${c}"; mkdir "$d"; /root/dyq/workspace/.tools/baselines/baseline-run "$b" python "$P/run_case.py" "$b" "$o" "$c" "$d"; }
run_one ops-math SumFwdOp mlp-intermediate-float32
run_one ops-nn BatchNormBwdOp resnet-stage3-float32
run_one ops-nn CountNonzeroFwdOp sparsity-seq-float16
run_one ops-nn SoftmaxFwdOp qwen3-8b-lm-head-b1-float32
run_one ops-nn VectorNormFwdOp inf-long-seq-bfloat16
