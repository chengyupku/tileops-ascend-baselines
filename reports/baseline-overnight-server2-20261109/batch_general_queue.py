"""Guarded serial engineering baseline verification; resume completed keys without rerun."""
import pathlib,subprocess,json,time,os
P=pathlib.Path(__file__).resolve().parent

def active_prior():
 for p in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
  try:parts=p.read_bytes().split(b'\0')
  except OSError:continue
  if any(str(P/name).encode() in parts for name in ('batch_w4.py','batch_follow.py')):return True
 return False
while active_prior():time.sleep(5)
log=P/'GENERAL_BATCH.json';rows=json.loads(log.read_text()) if log.exists() else []
def save():
 tmp=log.with_suffix('.tmp');tmp.write_text(json.dumps(rows,indent=2));tmp.replace(log)
def attempt(task,base):
 t=json.loads(task.read_text());root=P/'runs'/str(t['ordinal'])
 previous=sorted(root.glob(base+'-a*')) if root.exists() else []
 for out in previous:
  try:g=json.loads((out/'guard.json').read_text())
  except (OSError,ValueError):g={}
  if g.get('phase')=='finished' and g.get('performance_valid') and g.get('workload_exit_code')==0 and (out/'measurement.json').exists():return 0,str(out)
  if (out/'device_failure.json').exists() or (out/'prepare_failure.json').exists():return 1,str(out)
 label=base+'-a'+str(len(previous)+1)
 r=subprocess.run(['python3',str(P/'run_one_queue.py'),task.name,label,'device'])
 out=root/label
 return r.returncode,str(out)
for task in sorted((P/'tasks').glob('*-gated-solve.json')):
 ordinal=json.loads(task.read_text())['ordinal']
 if any(r['ordinal']==ordinal and r['status'] in ('measured_pending_raw_audit','deferred_after_both_candidates') for r in rows):continue
 rc,out=attempt(task,'gated-solve-v3')
 fallback=None
 if rc == 1:
  # Preserve precision failure; unchanged arithmetic/index-only route is a separate candidate.
  fallback=rc
  rc,out=attempt(P/'tasks'/f'{ordinal}-gated.json','gated-index-v3')
 rows.append({'ordinal':ordinal,'out':out,'exit':rc,'solve_rejected_exit':fallback,'status':'measured_pending_raw_audit' if rc==0 else ('deferred_after_both_candidates' if rc==1 else 'waiting_resource_or_nightly')});save()
 if rc not in (0,1):raise SystemExit(rc)
 if sum(r['status']=='measured_pending_raw_audit' for r in rows)%10==0:
  subprocess.run(['python3',str(P/'replay_raw.py')],check=True)
  subprocess.run(['python3',str(P/'package_evidence.py')],check=True)
print('general torch queue exhausted; inspect deferred items then native queue',flush=True)
