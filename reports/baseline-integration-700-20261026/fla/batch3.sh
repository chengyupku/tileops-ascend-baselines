#!/bin/bash
P=/root/dyq/workspace/reports/baseline-integration-700-20261026/fla
export PYTHONDONTWRITEBYTECODE=1
/root/dyq/workspace/.tools/baselines/baseline-run flash-linear-attention-npu python "$P/run_case.py" CumsumFwdOp hidden-state-float16 "$P/CumsumFwdOp.explicit.checker.json" > "$P/CumsumFwdOp.explicit.log" 2>&1
for cid in decode-beta-l2norm-bfloat16 q35-gva-decode-b1-float16 q35-35b-4096-float16; do
 /root/dyq/workspace/.tools/baselines/baseline-run flash-linear-attention-npu python "$P/run_gdn.py" GDNFwdOp "$cid" "$P/GDNFwdOp.$cid.checker.json" > "$P/GDNFwdOp.$cid.log" 2>&1
done
