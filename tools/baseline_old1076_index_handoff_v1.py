"""CPU-only finite boundary handoff and evidence-gated BF16 rollout."""
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
O=ROOT/'reports/baseline-old1076-20261029'
record=json.loads((O/'INDEX_HANDOFF.json').read_text())
def save():
 t=O/'INDEX_HANDOFF.tmp';t.write_text(json.dumps(record,indent=2)+'\n');os.replace(t,O/'INDEX_HANDOFF.json')
def previous_alive():
 try:
  s=Path('/proc/%s/stat'%record['previous_identity']['pid']).read_text().rsplit(')',1)[1].split()
  return int(s[19])==record['previous_identity']['start_ticks'] and s[0]!='Z'
 except FileNotFoundError:return False
record.update(handoff_pid=os.getpid(),phase='waiting_current_pair');save()
while previous_alive():time.sleep(1)
if (O/'STOP').exists() or (O/'server1/STOP').exists():raise RuntimeError('Explicit STOP present; leave stopped')
runner=ROOT/'tools/run_baseline_old1076_index_v1.py'
cmd=[sys.executable,str(runner),'--campaign',str(O),'--host','server1','--selection']
record.update(phase='pilot',pilot_started_at=time.time());save()
rc=subprocess.run(cmd+[str(O/'REUSE_BF16_PILOT.json')]).returncode
ids=set(json.loads((O/'REUSE_BF16_PILOT.json').read_text())['pairs']);latest={}
for line in (O/'server1/journal.jsonl').read_text().splitlines():
 r=json.loads(line)
 if r['id'] in ids:latest[r['id']]=r
passed=rc==0 and set(latest)==ids and all(r['status']=='cached' and json.loads((Path(r['measurement']).parent/'correctness.json').read_text()).get('status')=='verified' for r in latest.values())
record.update(pilot_exit_code=rc,pilot_passed=passed,pilot_results={k:r['status'] for k,r in latest.items()})
if not passed:
 record.update(phase='pilot_needs_review',finished_at=time.time());save();sys.exit(1)
record.update(phase='promoted',rollout_at=time.time());save()
os.execv(sys.executable,cmd+[str(O/'REUSE_BF16_QUEUE.json')])
