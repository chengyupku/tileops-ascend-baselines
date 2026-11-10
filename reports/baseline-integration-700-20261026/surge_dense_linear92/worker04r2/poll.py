from pathlib import Path
import json,subprocess,datetime,shlex
W=Path(__file__).resolve().parent
R='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/surge_dense_linear92/worker04r2'
SSH='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
code="from pathlib import Path;import json;w=Path("+repr(R)+");print(json.dumps(dict(heartbeat=json.loads((w/'HEARTBEAT.json').read_text()),lock=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text()),finished=[p.parent.name for p in w.glob('attempt*/GUARD_RESULT.json')])) )"
p=subprocess.run([SSH,'openlibing-DevEnv-055409','python3 -c '+shlex.quote(code)],text=True,capture_output=True,check=True)
r=json.loads(p.stdout);r['observed_local_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
with (W/'LIVE_GUARD_SNAPSHOTS.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
h=r['heartbeat'];m=r['lock'];(W/'HEARTBEAT.json').write_text(json.dumps(dict(utc=r['observed_local_utc'],ownership='exclusive SCOPE accepted',task_type='补测/实际对接',exact_key=['ops-transformer','GDNFwdOp',h['case']],stage=h['phase'],actual_lock_owner=m.get('owner'),actual_lock_phase=m.get('phase'),last_substantive_output='RESOURCE_BOUND.json and submitted b128/b256 checker jobs',blocker='FIFO queue' if m.get('owner')!='baseline700-20261026/surge_dense_linear92/worker04r2' else None,next='collect complete checker receipts',accepted_added=0),indent=2))
print(json.dumps(dict(utc=r['observed_local_utc'],remote=h,lock_owner=m.get('owner'),phase=m.get('phase'),finished=r['finished'])))
