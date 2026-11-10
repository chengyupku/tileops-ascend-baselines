#!/bin/bash
P=/root/dyq/workspace/reports/baseline-integration-700-20261026/fla
export PYTHONDONTWRITEBYTECODE=1
for pair in 'AddFwdOp linear-bias-float32' 'GemmFwdOp ds-v3-m64-down-bfloat16' 'GroupedGemmFwdOp qwen3-30b-ep8-empty-bfloat16' 'GroupedGemmFwdOp moe-fwd-bfloat16' 'GLAFwdOp decode-first-token-float16' 'DeltaNetFwdOp decode-continued-float16'; do
 read -r op cid <<< "$pair"
 /root/dyq/workspace/.tools/baselines/baseline-run flash-linear-attention-npu python "$P/run_case.py" "$op" "$cid" "$P/$op.$cid.checker.json" > "$P/$op.$cid.log" 2>&1
done
