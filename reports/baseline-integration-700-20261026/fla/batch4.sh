#!/bin/bash
P=/root/dyq/workspace/reports/baseline-integration-700-20261026/fla
export PYTHONDONTWRITEBYTECODE=1
for cid in qwen3-30b-ep8-empty-bfloat16 moe-fwd-bfloat16; do
 /root/dyq/workspace/.tools/baselines/baseline-run flash-linear-attention-npu python "$P/run_case.py" GroupedGemmFwdOp "$cid" "$P/GroupedGemmFwdOp.$cid.repair.checker.json" > "$P/GroupedGemmFwdOp.$cid.repair.log" 2>&1
 code=$?
 if [ "$code" -ne 0 ]; then break; fi
done
