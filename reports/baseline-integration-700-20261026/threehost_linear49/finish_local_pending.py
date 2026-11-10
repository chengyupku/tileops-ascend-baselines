from pathlib import Path
import subprocess,json,time,datetime
P=Path(__file__).resolve().parent
keys=[r['case_id'] for r in json.loads((P/'SCOPE.json').read_text()) if r['baseline']=='flash-linear-attention-npu' and r['operator']=='GDNFwdOp' and r['status']=='blocked']
for cid in keys:
 ds=[d for d in P.glob('fla_'+cid+'*') if d.is_dir()];done=False
 for d in ds:
  if (d/'GUARD_RESULT.json').exists() and (d/'checker.json').exists():
   g=json.loads((d/'GUARD_RESULT.json').read_text());c=json.loads((d/'checker.json').read_text())
   if g['returncode'] not in [75,76,78,124] and (c.get('status')=='passed' or 'Tensor-likes are not close' in c.get('error','')):done=True
 if done:continue
 label='continuation'+str(len(ds));p=subprocess.Popen(['python3',str(P/'local_fla_run.py'),cid,label]);last=0
 while p.poll() is None:
  if time.monotonic()-last>30:
   (P/'HEARTBEAT.local.json').write_text(json.dumps(dict(phase='exact_fla_after_external_exit',currentkey=cid,host='local',lastprogress=datetime.datetime.now(datetime.timezone.utc).isoformat(),blockedreason=None)));last=time.monotonic()
  time.sleep(1)
 d=json.loads((P/('fla_'+cid+'_'+label)/'GUARD_RESULT.json').read_text());print(cid,d['returncode'],flush=True)
 if d['returncode'] in [75,76,78]:break
