"""Expose closed cancelled attempts to the existing full-raw archive collector."""
import subprocess,shlex,json,pathlib,time
P=pathlib.Path(__file__).resolve().parent
code=r'''import pathlib,json,os,time
root=pathlib.Path('/opt/torch-baseline400-20261104/server1/runs'); rows=[]
def live(pid):
 return pid and pathlib.Path('/proc',str(pid)).exists()
for f in root.glob('*/a*/CANCELLED.json'):
 d=f.parent
 if (d/'DONE.json').exists():continue
 c=json.loads(f.read_text());gpath=d/'guard.json'
 if gpath.exists():
  g=json.loads(gpath.read_text())
  if g.get('phase')!='finished' or live(g.get('pid')) or live(g.get('child',{}).get('pid')):continue
 else:
  # CPU-only cancellation has no device guard; require an ended prepare and no device evidence.
  prep=c.get('prepare',{})
  if c.get('device') or 'end' not in prep or 'returncode' not in prep or (d/'profiles').exists():continue
 phases=[c.get('prepare',{}),c.get('device',{})]
 if any(live(x.get('pid')) for x in phases):continue
 receipt=dict(c,status='cancelled',archive_only=True,finalized_at=time.time(),reason='Closed cancelled attempt; preserve full raw; not accepted and no retry authorization')
 with (d/'DONE.json').open('x') as out:json.dump(receipt,out,indent=2)
 rows.append(str(d))
print(json.dumps(rows))'''
r=subprocess.run(['ssh','-T','-o','ClearAllForwardings=yes','openlibing-DevEnv-661495',shlex.join(['python3','-c',code])],capture_output=True,text=True,check=True,timeout=60)
(P/'server1/CANCELLED_ARCHIVE_FINALIZATION.json').write_text(json.dumps({'at':time.time(),'finalized':json.loads(r.stdout)},indent=2))
print(r.stdout)
