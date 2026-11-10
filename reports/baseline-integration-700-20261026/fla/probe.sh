#!/bin/bash
set +e
P=/root/dyq/workspace/reports/baseline-integration-700-20261026/fla
export PYTHONDONTWRITEBYTECODE=1
for pair in 'AddFwdOp hidden-state-prefill-float16' 'CumsumFwdOp hidden-state-float16' 'GDNFwdOp decode-b1-float16' 'GemmFwdOp ds-v3-m2-qkv-a-bfloat16' 'KDAFwdOp kda-tp1-16x8192-bfloat16'; do
 read -r op cid <<< "$pair"
 /root/dyq/workspace/.tools/baselines/baseline-run flash-linear-attention-npu python "$P/run_case.py" "$op" "$cid" "$P/$op.checker.json" > "$P/$op.log" 2>&1
 done
