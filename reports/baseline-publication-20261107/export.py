import json,hashlib,math,statistics,datetime,re,tarfile,yaml
from pathlib import Path
P=Path(__file__).resolve().parent; ROOT=P.parent.parent; S=P/'site'; T=ROOT/'reports/torch-v010-20261106'
def read(p):return json.loads(p.read_text())
hash_cache_path=P/'file-hashes.json'
hash_cache=json.loads(hash_cache_path.read_text()) if hash_cache_path.exists() else {}
def sha(p):
 st=p.stat(); token=[st.st_size,st.st_mtime_ns]
 prior=hash_cache.get(str(p))
 if prior and prior[:2]==token:return prior[2]
 with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
 hash_cache[str(p)]=token+[digest]
 hash_cache_path.write_text(json.dumps(hash_cache))
 return digest
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
def canon(x):
 if isinstance(x,float) and not math.isfinite(x):return 'NaN' if math.isnan(x) else ('Infinity' if x>0 else '-Infinity')
 if isinstance(x,str) and x in ('inf','-inf'):return 'Infinity' if x=='inf' else '-Infinity'
 if isinstance(x,dict):return {k:canon(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [canon(v) for v in x]
 return x
def field(name,text,last=False):
 ms=list(re.finditer(r'"'+re.escape(name)+r'"\s*:\s*',text));assert ms,name
 return json.JSONDecoder().raw_decode(text[(ms[-1] if last else ms[0]).end():])[0]
def metadata(p):
 with p.open() as f:
  prefix=f.read(65536);f.seek(max(0,p.stat().st_size-131072));suffix=f.read()
 m={k:field(k,prefix) for k in ['key','task_id','status','latency_ms','latency_samples_ms']}
 m['correctness_sha256']=field('correctness_sha256',prefix if '"correctness_sha256"' in prefix else suffix)
 m['parser_revision']=field('parser_revision',suffix.split('"measurement_protocol"')[0],last=True)
 m['protocol']=field('measurement_protocol',suffix,last=True)
 if isinstance(m['protocol'],dict):m['protocol']=m['protocol']['id']
 if m['key'][0]=='torch-eager':
  m.update({k:field(k,suffix,last=True) for k in ['source_sha256','warmup','repeats','inputs_preserved']});m['hardware']=field('hardware',prefix)
 else:
  m['identity']=field('identity',suffix);m['experiment_v3_sources']=field('experiment_v3_sources',suffix)
 return m
# Freeze receipts first; producer is free to continue, this release has a fixed cutoff.
ledgers={camp:{h:read(P/f'{camp}-{h}-frozen.json') for h in ['server1','server2']} for camp in ['torch-v010-20261106','baseline-flex-20261106']}
archives={};preflight={}
for h,ledger in ledgers['torch-v010-20261106'].items():
 for tid,r in ledger.items():
  assert r['accepted'] and r['verified'];archives[r['archive']]=r['sha256']
  if r.get('archive_member'):preflight.setdefault(r['archive'],set()).add(r['archive_member'])
for ledger in ledgers['baseline-flex-20261106'].values():
 for r in ledger.values():
  if r.get('status')=='accepted':
   a=r['archive'];assert a['accepted'] and a['verified'] and not a['error'];archives[a['local']]=a['sha256']
cache=P/'archive-hashes.json'
verified=read(cache) if cache.exists() else {}
for i,(name,digest) in enumerate(archives.items()):
 stat=Path(name).stat(); stamp=[digest,stat.st_size,stat.st_mtime_ns]
 if verified.get(name)!=stamp:
  assert sha(Path(name))==digest,name
  verified[name]=stamp
  save(cache,verified)
 if i%100==0:print('verified archive',i,'/',len(archives),flush=True)
pre=P/'preflight-metadata';pre.mkdir(exist_ok=True)
for i,(name,members) in enumerate(preflight.items()):
 dest=pre/str(i);dest.mkdir(exist_ok=True)
 if (dest/'EXTRACTED.json').exists():
  assert read(dest/'EXTRACTED.json')=={'archive':name,'sha256':archives[name]}
  preflight[name]=dest
  continue
 with tarfile.open(name,'r|gz') as tar:
  for m in tar:
   parts=Path(m.name).parts
   if len(parts)==2 and parts[0] in members and parts[1] in ['measurement.json','task.json','correctness.json','guard.json','prepare.json','device.log']:
    out=dest/m.name;out.parent.mkdir(exist_ok=True);out.write_bytes(tar.extractfile(m).read())
 save(dest/'EXTRACTED.json',{'archive':name,'sha256':archives[name]})
 preflight[name]=dest
families={}
for f in (T/'source/src/tileops/manifest/spec').glob('*.yaml'):
 for op,spec in (yaml.safe_load(f.read_text()) or {}).items():
  if op.endswith('Op'):families[op]=spec.get('family',f.stem)
old=read(S/'data/baseline-cache/snapshot.json');save(P/'snapshot.before.json',old)
for r in old['rows']:families.setdefault(r['op'],r['workload']['family'])
current={tuple(r['task']['key']):canon(r['task']['case']) for r in read(T/'QUEUE.json')['tasks']}
proofs={sha(f) for f in T.glob('SOURCES.v*.json')};updates=[];audit=[]
for camp,hosts in ledgers.items():
 for host,ledger in hosts.items():
  for tid,r in ledger.items():
   if camp.startswith('baseline') and r.get('status')!='accepted':continue
   if r.get('archive_member'):
    d=preflight[r['archive']]/r['archive_member'];arc=r['archive'];a=r
   elif camp.startswith('torch'):
    d=Path(r['archive']).parent;arc=r['archive'];a=read(d/'ARCHIVE.json')
   else:
    d=Path(r['receipt']).parent;arc=r['archive']['local'];a=read(d/'ARCHIVE.json')
   m=metadata(d/'measurement.json');t=read(d/'task.json');c=read(d/'correctness.json');g=read(d/'guard.json');key=tuple(m['key'])
   assert tuple(t['key'])==key and tid==m['task_id']
   assert g['performance_valid'] and g['phase']=='finished' and g['exit_code']==g['workload_exit_code']==0
   assert c['status']=='verified' and c['inputs_preserved'] and sha(d/'correctness.json')==m['correctness_sha256']
   check=c.get('check',c.get('verification'));assert check['checked_outputs']>0 and check['unchecked_reason'] is None, (key,check)
   assert m['status']=='measured_pending_guard' and m['protocol']=='ascend950-isolated-ffts-v3' and m['parser_revision']=='v3-task-coverage-full32-20261103', (key,m['status'],m['protocol'],m['parser_revision'])
   samples=m['latency_samples_ms'];assert len(samples)==15 and all(math.isfinite(v) and v>0 for v in samples)
   assert math.isclose(statistics.mean(samples),m['latency_ms'],rel_tol=1e-12)
   work=canon(t['case'])
   if camp.startswith('torch'):
    assert work==current[key];assert m['source_sha256'] in proofs and (m['hardware']['cube'],m['hardware']['vector'])==(32,64)
    assert m['warmup']==3 and m['repeats']==15 and m['inputs_preserved']
    assert a['measurement']['raw_windows']==15 and math.isclose(a['measurement']['latency_ms'],m['latency_ms'],rel_tol=1e-12)
   else:
    assert not c.get('provider_source_changed') and not c.get('oracle_source_changed')
    env=m['identity']['environment'];assert (env['aicore_count'],env['vectorcore_count'])==(32,64)
    assert m['identity']['protocol']['id']=='ascend950-isolated-ffts-v3'
   work['family']=work.get('family',families.get(key[1]));assert work['family'],key
   measurement_id=sha(d/'measurement.json')
   if (d/'FILES.json').exists():assert read(d/'FILES.json')['files']['measurement.json']['sha256']==measurement_id
   row=dict(library=key[0],op=key[1],case=key[2],latency_us=m['latency_ms']*1000,samples=15,measurement_id=measurement_id,protocol=m['protocol'],parser_revision=m['parser_revision'],workload=work,tileops_latency_us=None,ratio=None)
   updates.append(row);audit.append(dict(key=key,archive=arc,archive_sha256=archives[arc],measurement_id=measurement_id,campaign=camp,guard_owner=g['owner'],correctness_check=check,correctness_policy=c.get('policy')))
   if len(updates)%100==0:print('validated measurement',len(updates),flush=True)
assert sum(x['library']=='torch-eager' for x in updates)==234
# Current view prefers the newly measured torch contract, while historical contracts remain available to CI.
key=lambda r:(r['library'],r['op'],r['case']);case=lambda r:(r['op'],r['case']);contract=lambda r:json.dumps(canon(r['workload']),sort_keys=True)
rows={key(r):r for r in old['rows']};historical=list(old.get('historical_rows',[]));changed=[]
for r in updates:
 k=key(r);prior=rows.get(k)
 if prior:
  historical.append(prior)
  if contract(prior)==contract(r):
   r.update({f:v for f,v in prior.items() if f.startswith('tileops_')})
   if r['tileops_latency_us'] is not None:r['ratio']=r['latency_us']/r['tileops_latency_us']
  else:changed.append(k)
 rows[k]=r
preferred={case(r):contract(r) for r in updates if r['library']=='torch-eager'}
for k,r in list(rows.items()):
 if case(r) in preferred and contract(r)!=preferred[case(r)]:historical.append(rows.pop(k))
# A native remeasurement can retain an old contract; never silently overwrite latest torch.
seen=set();history=[]
for r in reversed(historical):
 ident=(key(r),contract(r))
 if ident in seen or (key(r) in rows and contract(rows[key(r)])==contract(r)):continue
 seen.add(ident);history.append(r)
# Apply a case's still-compatible TileOPs value consistently to all its baselines.
tiles={case(r):r for r in old['rows'] if r.get('tileops_latency_us') is not None}
for r in rows.values():
 prior=tiles.get(case(r))
 if prior and contract(prior)==contract(r):
  r.update({f:v for f,v in prior.items() if f.startswith('tileops_')});r['ratio']=r['latency_us']/r['tileops_latency_us']
 else:
  for f in list(r):
   if f.startswith('tileops_'):del r[f]
  r.update(tileops_latency_us=None,ratio=None)
result={**old,'rows':list(rows.values()),'historical_rows':history,'frozen_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'manifest_cases':len({case(r) for r in rows.values()}),'baseline_release':'verified-20261107','publication_notes':['2026-11-07: publish 234 strictly accepted v0.1.0 torch measurements and 400 accepted native retests. Unchanged saved baselines retain their original protocol labels.','The displayed inventory retains previously published workloads; it is not a claim of complete v0.1.0 coverage. Previous incompatible contracts are preserved in historical_rows for exact-contract CI matching.','All new archive hashes, correctness and valid 32AIC/64AIV guard receipts verified. Fifteen-window raw replay is established by the campaign acceptance receipts.']}
result['tileops_rows']=[r for r in old.get('tileops_rows',[]) if case(r) not in {case(x) for x in rows.values()}]
save(S/'data/baseline-cache/snapshot.json',result)
release={'schema':2,'id':'verified-20261107','count':len(rows),'snapshot_sha256':sha(S/'data/baseline-cache/snapshot.json'),'measurements':[dict(library=r['library'],op=r['op'],case=r['case'],measurement_id=r['measurement_id'],protocol=r['protocol']) for r in rows.values()]}
save(S/'data/baseline-cache/latest-release.json',release)
save(P/'AUDIT.json',{'new_torch':234,'new_native':len(updates)-234,'rows':len(rows),'torch_rows':sum(r['library']=='torch-eager' for r in rows.values()),'historical_contract_rows':len(history),'changed_contract_keys':changed,'snapshot_sha256':release['snapshot_sha256'],'evidence':audit,'raw_replay':'Relies on admitted campaign replay receipts; this publication rehashes archives and metadata, not a second raw decode.'})
print('DONE', {k:v for k,v in read(P/'AUDIT.json').items() if k not in ['evidence','changed_contract_keys']},flush=True)
