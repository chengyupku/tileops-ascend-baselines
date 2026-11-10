"""Publish accepted campaign receipts; no device access or replacement of other libraries."""
import json,pathlib,hashlib,datetime,math,statistics,sys
P=pathlib.Path(__file__).resolve().parents[1];OUT=pathlib.Path(__file__).parent;site=pathlib.Path(sys.argv[1])
def read(p):return json.loads(p.read_text())
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,separators=(',',':'))+'\n')
def canonical(x):
 if isinstance(x,float) and not math.isfinite(x):return 'NaN' if math.isnan(x) else ('Infinity' if x>0 else '-Infinity')
 if isinstance(x,str) and x in ('inf','-inf'):return 'Infinity' if x=='inf' else '-Infinity'
 if isinstance(x,dict):return {k:canonical(v) for k,v in x.items()}
 if isinstance(x,list):return [canonical(v) for v in x]
 return x
source=site/'data/baseline-cache/snapshot.json';old=read(source);save(OUT/'snapshot.before.json',old)
existing={tuple(r[k] for k in ['library','op','case']):r for r in old['rows']};bycase={(r['op'],r['case']):r for r in old['rows']};proofs={sha(f):f.name for f in P.glob('SOURCES*.json')};rows=[];audit=[]
for host in ['server1','server2']:
 progress=read(P/host/'PROGRESS.json');plan={r['id']:r for r in read(P/host/'PLAN.json')['rows']};assert progress.keys()==plan.keys()
 for tid,r in progress.items():
  assert r['accepted'] and r['verified'] and not r['error'] and r['measurement']['raw_windows']==15,tid
  archive=pathlib.Path(r['archive']);d=archive.parent
  assert archive.stat().st_size==r['bytes'] and sha(archive)==r['sha256'],tid
  t=read(d/'task.json');c=read(d/'correctness.json');g=read(d/'guard.json');m=read(d/'measurement.json');prep=read(d/'prepare.json')
  assert t==plan[tid] and m['task_id']==tid and m['key']==t['key']==r['measurement']['key']
  assert canonical(prep['params'])==canonical(t['case']['params']) and c['status']=='verified' and c['inputs_preserved'] and m['inputs_preserved']
  assert c['check']['unchecked_reason'] is None and c['check']['checked_outputs']>0
  if c['check']['checked_outputs']!=c['check']['total_outputs']:
   # Frozen checker validates optional None output identity but counts tensors only.
   allowed={'14bf100f1b1fe2e02e5d':(1,3),'5d380863728c68305b2a':(1,2)}
   assert (c['check']['checked_outputs'],c['check']['total_outputs'])==allowed[tid]
   assert not {'running_mean','running_var','shared_w_gate_up','shared_w_down'} & t['case']['tensors'].keys()
  assert sha(d/'correctness.json')==m['correctness_sha256'] and m['source_sha256'] in proofs and c['source_sha256']==m['source_sha256']
  assert g['performance_valid'] and g['phase']=='finished' and g['workload_exit_code']==g['exit_code']==0 and g['owner']=='torch1135/'+host+'/'+tid
  assert (m['hardware']['cube'],m['hardware']['vector'])==(32,64)
  assert len(m['timing']['retained_profiles'])==len(m['timing']['latency_samples_ms'])==15
  assert math.isclose(statistics.mean(m['timing']['latency_samples_ms']),r['measurement']['latency_ms'],rel_tol=1e-12)
  assert all(x['parser_revision'] in {'v3-task-coverage-full32-20261103','v3-task-coverage-20261103'} for x in m['timing']['retained_profiles']),m['timing']['retained_profiles'][0]['parser_revision']
  library,op,case=t['key'];assert library=='torch-eager';us=r['measurement']['latency_ms']*1000;assert math.isfinite(us) and us>0
  prior=bycase.get((op,case))
  if prior:assert prior['workload']==canonical(t['case']),(op,case,'contract conflict')
  row=dict(library=library,op=op,case=case,latency_us=us,samples=15,measurement_id=sha(d/'measurement.json'),protocol='ascend950-isolated-ffts-v3',parser_revision=m['timing']['retained_profiles'][0]['parser_revision'],workload=canonical(t['case']),tileops_latency_us=None,ratio=None)
  if prior:
   for k,v in prior.items():
    if k.startswith('tileops_'):row[k]=v
   if row['tileops_latency_us'] is not None:row['ratio']=us/row['tileops_latency_us']
  notes=[]
  if (d/'new_reference.json').exists():notes.append(read(d/'new_reference.json'))
  if (d/'interface_adaptation.json').exists():notes.append(read(d/'interface_adaptation.json'))
  rows.append(row);audit.append(dict(task_id=tid,host=host,attempt=d.name,archive=str(archive),archive_sha256=r['sha256'],measurement_id=row['measurement_id'],source_manifest=proofs[m['source_sha256']],adaptations=notes))
  if len(rows)%100==0:print('validated',len(rows),flush=True)
assert len(rows)==1135 and len({(r['op'],r['case']) for r in rows})==1135
replaced=sum((r['library'],r['op'],r['case']) in existing for r in rows)
for row in rows:existing[(row['library'],row['op'],row['case'])]=row
result={**old,'frozen_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rows':list(existing.values())}
result['publication_notes']=['2026-11-04: added 1135 torch-eager measurements, full correctness, physical 32 AI Cores, 15 retained raw profiler windows and verified archives. New torch rows use v3 device-task timing; other libraries retain their recorded protocols.','DeltaNetChunk uses newly authorized six-output torch device references checked against the unchanged independent CPU oracle.','GQA qwen35-9b-rope-float16 uses CPU generation of parameter-only inverse frequencies. Position, trigonometry, rotation and attention execute on NPU. Reported device time excludes host parameter preparation and is not end-to-end latency.']
save(source,result)
public={'schema':1,'campaign':'torch-baseline1135-20261103','count':1135,'snapshot_sha256':sha(source),'measurements':[{'op':r['op'],'case':r['case'],'measurement_id':r['measurement_id'],'protocol':r['protocol']} for r in rows]}
save(site/'data/baseline-cache/torch1135-release.json',public)
save(OUT/'AUDIT.json',{'count':1135,'added':1135-replaced,'replaced':replaced,'total_rows':len(result['rows']),'snapshot_sha256':sha(source),'rows':audit})
print('DONE',1135-replaced,replaced,len(result['rows']),flush=True)
