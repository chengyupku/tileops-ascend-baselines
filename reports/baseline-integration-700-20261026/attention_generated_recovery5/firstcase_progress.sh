R=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_generated_recovery5
tail -12 "$R/gpt-oss-sink-decode-bfloat16-bfloat16.log"
test ! -e "$R/gpt-oss-sink-decode-bfloat16-bfloat16.checker.json" || python3 - <<'PY'
import json,pathlib
p=pathlib.Path('/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_generated_recovery5/gpt-oss-sink-decode-bfloat16-bfloat16.checker.json');d=json.loads(p.read_text());print({k:d.get(k) for k in ['status','stage','error','verification']})
PY
