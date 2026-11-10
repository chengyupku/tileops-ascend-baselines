set -e
R=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_generated_recovery5
for pair in \
 'GQAPagedFwdOp gpt-oss-sink-mixed-bfloat16-bfloat16' \
 'GQAPagedFwdOp llama-8b-sliding-window-float16-float16' \
 'GQAVarlenFwdOp gpt-oss-sink-mixed-bfloat16' \
 'GQAVarlenFwdOp gpt-oss-sink-mixed-float16'; do
 read -r op key <<< "$pair"
 bash "$R/RUN_CASE.sh" "$op" "$key"
 python3 - "$R/$key.checker.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]));assert r['status']=='verified' and r['inputs_preserved'] and r['verification']['checked_outputs']==r['verification']['total_outputs']>0
PY
done
