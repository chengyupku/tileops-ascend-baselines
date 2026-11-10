import json,hashlib,math,statistics,datetime,sys,yaml,re
from pathlib import Path
P=Path('/root/dyq/workspace/reports/torch-all2208-20261105');O=Path('/root/dyq/workspace/reports/baseline-publication-20261106');S=Path('/tmp/tileops-site-torch2208-20261106')
def read(p):return json.loads(p.read_text())
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def measurement_metadata(d):
 p=d/'measurement.json'
 files=read(d/'FILES.json')['files'];assert p.stat().st_size==files['measurement.json']['size']
 with p.open('rb') as f:
  prefix=f.read(65536).decode();f.seek(max(0,p.stat().st_size-65536));suffix=f.read().decode()
 def field(name, text, last=False):
  matches=list(re.finditer(r'"'+re.escape(name)+r'"\s*:\s*',text));assert matches,name
  match=matches[-1] if last else matches[0]
  return json.JSONDecoder().raw_decode(text[match.end():])[0]
 m={k:field(k,prefix) for k in ['status','key','task_id','hardware']}
 m.update({k:field(k,suffix,last=True) for k in ['correctness_sha256','source_sha256','warmup','repeats','inputs_preserved']})
 timing={k:field(k,prefix) for k in ['latency_ms','latency_samples_ms']}
 timing['measurement_protocol']=field('measurement_protocol',suffix,last=True)
 assert timing['measurement_protocol']=='ascend950-isolated-ffts-v3'
 # The admitted archive receipt and final audit establish all 15 raw-window replays.
 # Read the last retained profile's parser label without loading its millions of tasks.
 revisions=re.findall(r'"parser_revision"\s*:\s*"([^"\n]+)"',suffix)
 assert revisions and set(revisions)<={'v3-task-coverage-20261103','v3-task-coverage-full32-20261103'}
 timing['retained_profiles']=[{'parser_revision':revisions[0]}]*15
 m['timing']=timing;m['file_sha256']=files['measurement.json']['sha256']
 return m
def canon(x):
 if isinstance(x,float) and not math.isfinite(x):return 'NaN' if math.isnan(x) else ('Infinity' if x>0 else '-Infinity')
 if isinstance(x,str) and x in ('inf','-inf'):return 'Infinity' if x=='inf' else '-Infinity'
 if isinstance(x,dict):return {k:canon(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [canon(v) for v in x]
 return x
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
audit=read(Path(read(P/'COMPLETION.json')['final_audit']));current={tuple(x['key']):x for x in read(P/'CURRENT_CASES.json')['rows']};assert len(current)==len(audit['rows'])==2208
families={}
for f in (P/'source/src/tileops/manifest/spec').glob('*.yaml'):
 for op,spec in (yaml.safe_load(f.read_text()) or {}).items():
  if op.endswith('Op'):families[op]=spec.get('family',f.stem)
source=S/'data/baseline-cache/snapshot.json';old=read(source);(O/'snapshot.before.json').write_bytes(source.read_bytes());bycase={(x['op'],x['case']):x for x in old['rows']};proofs={sha(f):f.name for f in P.glob('SOURCES.v*.json')};rows=[];evidence=[]
for i,r in enumerate(audit['rows']):
 archive=Path(r['archive']);d=archive.parent;assert sha(archive)==r['sha256'] and archive.stat().st_size==r['bytes']
 a=read(d/'ARCHIVE.json');m=measurement_metadata(d);t=read(d/'task.json');c=read(d/'correctness.json');g=read(d/'guard.json');prep=read(d/'prepare.json')
 key=tuple(t['key']);assert key==tuple(r['key'])==tuple(m['key'])==tuple(a['measurement']['key']);assert canon(t['case'])==canon(current[key]['case'])
 assert a['accepted'] and a['verified'] and not a['error'] and a['sha256']==r['sha256'] and a['measurement']['raw_windows']==15
 assert g['performance_valid'] and g['phase']=='finished' and g['exit_code']==g['workload_exit_code']==0
 assert c['status']=='verified' and c['inputs_preserved'] and m['inputs_preserved'] and c['check']['unchecked_reason'] is None and c['check']['checked_outputs']>0
 assert sha(d/'correctness.json')==m['correctness_sha256'] and m['source_sha256'] in proofs
 assert canon(prep['params'])==canon(t['case']['params']) and (m['hardware']['cube'],m['hardware']['vector'])==(32,64)
 samples=m['timing']['latency_samples_ms'];profiles=m['timing']['retained_profiles'];assert len(samples)==len(profiles)==15 and m['warmup']==3 and m['repeats']==15
 assert math.isclose(statistics.mean(samples),m['timing']['latency_ms'],rel_tol=1e-12)
 assert math.isclose(a['measurement']['latency_ms'],m['timing']['latency_ms'],rel_tol=1e-12)
 lib,op,case=key;us=m['timing']['latency_ms']*1000;assert math.isfinite(us) and us>0 and lib=='torch-eager'
 work=canon(t['case']);work['family']=families[op]
 row=dict(library=lib,op=op,case=case,latency_us=us,samples=15,measurement_id=m['file_sha256'],protocol='ascend950-isolated-ffts-v3',parser_revision=profiles[0]['parser_revision'],workload=work,tileops_latency_us=None,ratio=None)
 prior=bycase.get((op,case))
 if prior and canon(prior['workload'])==work:
  row.update({k:v for k,v in prior.items() if k.startswith('tileops_')})
  if row['tileops_latency_us'] is not None:row['ratio']=us/row['tileops_latency_us']
 rows.append(row);evidence.append(dict(key=key,archive=str(archive),archive_sha256=r['sha256'],measurement_id=row['measurement_id'],source_manifest=proofs[m['source_sha256']],correctness_checked=c['check']['checked_outputs'],correctness_total=c['check']['total_outputs']))
 if (i+1)%400==0:print('validated',i+1,flush=True)
assert {tuple([x['library'],x['op'],x['case']]) for x in rows}==set(current)
new={(x['op'],x['case']):x for x in rows};native=[x for x in old['rows'] if x['library']!='torch-eager'];conflicts=[x for x in native if (x['op'],x['case']) in new and canon(x['workload'])!=new[x['op'],x['case']]['workload']]
save(O/'native-contract-conflicts.json',conflicts);assert not conflicts,[(x['op'],x['case']) for x in conflicts][:10]
result={**old,'manifest_cases':2208,'frozen_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rows':native+rows}
result['publication_notes']=['2026-11-06: replaced the complete torch-eager baseline set with 2208 verified current workloads across 205 operators. Physical 32 AI Cores / 64 Vector Cores; must_keep_origin_dtype, HF32 disabled; v3 device timing, 3 warmups and 15 raw profiler windows.','All 2208 archives and correctness/guard receipts were verified for publication. Other libraries retain their recorded measurements and v2/v3 protocol labels.','Reported times measure device work, excluding CPU input and parameter preparation. Independent reference routes and adaptations are retained in the campaign evidence.']
save(source,result)
release={'schema':1,'campaign':'torch-all2208-20261105','count':2208,'snapshot_sha256':sha(source),'measurements':[{'op':x['op'],'case':x['case'],'measurement_id':x['measurement_id'],'protocol':x['protocol']} for x in rows]}
save(S/'data/baseline-cache/torch2208-release.json',release)
save(O/'AUDIT.json',{'count':2208,'native_preserved':len(native),'old_torch_replaced':sum(x['library']=='torch-eager' for x in old['rows']),'total_rows':len(result['rows']),'snapshot_sha256':sha(source),'rows':evidence,'verification':'Archive SHA rehashed; metadata and 15-sample mean checked; measurement IDs from FILES bound by admitted archives. Raw replay coverage relies on collector plus final ALL2208 audit; no second raw decoding claimed.'})
print('DONE',len(result['rows']),sha(source),flush=True)
