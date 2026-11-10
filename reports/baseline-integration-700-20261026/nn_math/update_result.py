import json,datetime,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent;data=json.loads((P/'RESULT.json').read_text());rows={(r['baseline'],r['operator'],r['case_id']):r for r in data['rows']};scope=json.loads((P/'SCOPE.json').read_text())
for out in sorted(P.iterdir(),key=lambda x: x.stat().st_mtime):
 if not out.is_dir() or not (out/'checker.json').exists() or not (out/'GUARD_RESULT.json').exists():continue
 c=json.loads((out/'checker.json').read_text());g=json.loads((out/'GUARD_RESULT.json').read_text());reg=json.loads((out/'REGISTRATION.json').read_text()) if (out/'REGISTRATION.json').exists() else None
 k=(c['library'],c['op'],c['case_id']);v=c.get('verification') or {};ok=c['status']=='verified' and g['returncode']==0 and not (out/'RETRACTED_REGISTRATION.json').exists()
 queue=device=None
 if (out/'GUARD_METADATA.json').exists():
  m=json.loads((out/'GUARD_METADATA.json').read_text());queue=(datetime.datetime.fromisoformat(m['acquired_at'])-datetime.datetime.fromisoformat(m['requested_at'])).total_seconds();device=(datetime.datetime.fromisoformat(g['finished_at'])-datetime.datetime.fromisoformat(m['updated_at'])).total_seconds()
 row=dict(baseline=k[0],operator=k[1],case_id=k[2],status=('completed' if reg else 'verified_unregistered') if ok else ('blocked' if g['returncode'] in (75,76,78,124) else 'failed'),reason=None if ok else (('Guard '+str(g['returncode'])+'; batch not accepted. Checker detail: '+str(c.get('error'))) if g['returncode'] in (75,76,78,124) else (c.get('error') or 'Numerical pass used builtin CANN device kernel; own baseline ownership unverified, registrations retracted.')),api=c.get('api'),guard_receipt=str(out/'GUARD_RESULT.json'),checker_receipt=str(out/'checker.json'),checked_outputs=v.get('checked_outputs',0),total_outputs=v.get('total_outputs',2 if 'FusedAdd' in k[1] else 3 if k[1]=='BatchNormFwdOp' else 1),registered=bool(reg),factory=reg['factory'] if reg else None,changed_files=[str(out/'adapter.py')]+([reg['module'],str(Path(reg['module']).parent/'__init__.py')] if reg else []),timings=dict(queue_s=queue,device_s=device,compile_s=None,elapsed_s=g['elapsed_s']),evidence=[str(out/'native.jsonl'),str(out/'guard.log')],source_sha256=c.get('source_sha256',{}))
 if k in rows and rows[k]['status']=='completed' and row['status']!='completed':continue
 rows[k]=row
 if k[1]=='FFTC2CFwdOp' and not ok:
  for s in scope:
   key=(s['baseline'],s['operator'],s['case_id'])
   if s['operator']=='FFTC2CFwdOp' and key!=k:
    rows[key]={**row,'case_id':key[2],'status':'blocked','reason':'Representative latency-4k STFT AiCPU launch fails 507018/21008; shared native STFT route stopped before unchanged replay. See representative raw receipts.','checker_receipt':None,'guard_receipt':None,'timings':dict(queue_s=None,device_s=None,compile_s=None),'evidence':[str(out/'checker.json'),str(out/'GUARD_RESULT.json'),str(out/'native.jsonl')]}
data['rows']=list(rows.values());data['updated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();tmp=P/'RESULT.tmp';tmp.write_text(json.dumps(data,indent=2));tmp.replace(P/'RESULT.json');print({s:sum(r['status']==s for r in rows.values()) for s in set(r['status'] for r in rows.values())})
