#!/bin/bash
# Future parent-owned validation after serial publication. Never run by this worker.
set -eu
CASE_ID=${1:?case id required}
RESULT_ROOT=${2:?fresh absolute output root required}
case "$CASE_ID" in moe-dgrad-float16|moe-fwd-float16) ;; *) exit 2;; esac
case "$RESULT_ROOT" in /*) ;; *) echo 'Absolute output root required' >&2; exit 2;; esac
P=/root/dyq/workspace/reports/baseline-integration-700-20261026/fp16_grouped_registration2
export FP16_GROUPED_GUARD_OWNER="baseline700/fp16_grouped_registration2/$CASE_ID"
export FP16_GROUPED_RUN_ROOT="$RESULT_ROOT"
exec npu-guard run --owner "$FP16_GROUPED_GUARD_OWNER" --wait 600 --timeout 900 -- \
 bash "$P/run_environment.sh" /root/dyq/workspace/.venv/bin/python -B \
 /root/dyq/workspace/TileOPs/scripts/ascend_pair_check.py \
 --op GroupedGemmFwdOp --case-id "$CASE_ID" --library ops-transformer --output "$RESULT_ROOT/representative.json"
