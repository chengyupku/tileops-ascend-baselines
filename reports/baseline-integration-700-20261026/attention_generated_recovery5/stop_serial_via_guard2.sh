python3 - <<'PY'
import json,os,signal,pathlib,datetime
r=pathlib.Path('/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_generated_recovery5')
m=json.loads(pathlib.Path('/run/lock/ascend-npu/device-0.lock').read_text())
assert m['owner']=='baseline700-20261026/attention_generated_recovery5' and m['phase']=='running'
pid=m['pid']; cmd=pathlib.Path(f'/proc/{pid}/cmdline').read_bytes().decode().replace('\0',' ')
assert 'npu_guard.py' in cmd and 'attention_generated_recovery5' in cmd
r.joinpath('serial_interrupt.json').write_text(json.dumps({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reason':'560 generated aiv targets; observed serial throughput cannot finish within 600s; change private supported make concurrency from 1 to 8','metadata':m,'cmdline':cmd},indent=2))
os.kill(pid,signal.SIGTERM)
PY
