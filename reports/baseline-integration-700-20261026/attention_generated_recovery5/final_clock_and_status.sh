python3 - <<'PY'
import os,json,pathlib,datetime
r=pathlib.Path('/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_generated_recovery5');d={'clock_ticks_per_second':os.sysconf('SC_CLK_TCK'),'at':datetime.datetime.now(datetime.timezone.utc).isoformat()};(r/'CLOCK_METADATA.json').write_text(json.dumps(d));print(d)
PY
npu-guard status --json > /root/dyq/workspace/reports/baseline-integration-700-20261026/attention_generated_recovery5/FINAL_GUARD_STATUS.json
