from pathlib import Path
import json,hashlib,subprocess,datetime,fcntl,os,shutil
W=Path('/root/dyq/workspace');D=W/'reports/baseline-integration-700-20261026';O=D/'sgl_attention_recovery32';P=D/'sgl_attention_register10';RW=Path('/opt/tileops-all-pairs/workspace');RP=RW/P.relative_to(W);read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,j):
 p=Path(p);t=p.with_suffix('.tmp-parent');t.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n');os.replace(t,p)
for p,h in read(P/'FINAL_HASHES.json').items():assert sha(P/p)==h,p
remote=read(P/'REMOTE_HASH_REQUEST.json');reg=read(P/'REGISTRATION.json');remote[reg['shared_module']]=sha(P/'stage/integration700_sgl_attention_register10.py');remote[str(RP/'dsa_adapter.py')]=sha(O/'interim14/dsa_adapter.py');remote[str(RP/'PINNED_LIBRARY.json')]=sha(P/'PINNED_LIBRARY.json')
for x in read(P/'PINNED_LIBRARY.json'):remote[x['path']]=x['sha256'];remote[x['resolved']]=x['sha256']
manifest={};proofs={}
for n in ['delta03','delta07']:
 F=O/n
 for p,h in read(F/'EVIDENCE_HASHES.json').items():assert sha(O/p)==h,p;manifest[p]=h
 assert sha(F/'dsa_adapter.py')==sha(O/'interim14/dsa_adapter.py')
 for x in read(F/'NATIVE_PROOF.json')['rows']:proofs[tuple(x['key'][f] for f in ('baseline','operator','case_id'))]=x
rows=[];events=0;missing=[];params={}
for r in read(P/'RECOMMENDED_ACCEPT.json')['rows']:
 c=read(r['checker']);g=read(r['guard']);k=r['key'];v=c['verification'];assert c['status']=='verified' and g['exit_code']==0 and v['checked_outputs']==v['total_outputs']==1 and not v['unchecked_reason'];assert c['inputs_preserved'] and not c['oracle_source_changed'] and not any(c.get(x) for x in ('error','sync_error','cleanup_error'));assert c['verifier_sha256']==sha(W/'TileOPs/scripts/ascend_pair_check.py');assert c['input_seed']==int.from_bytes(hashlib.sha256((k[1]+'::'+k[2]).encode()).digest()[:4],'big');params[k[2]]=c['case_snapshot']['parameters']
 for f in ('checker','guard'):assert sha(r[f])==manifest[str(Path(r[f]).relative_to(O))]
 for p,h in c['source_sha256'].items():
  if p.endswith('dsa_adapter.py'):assert h==sha(O/'interim14/dsa_adapter.py')
 for p,h in c['oracle_source_sha256'].items():remote[str(Path(p) if Path(p).is_absolute() else RW/p)]=h
 remote[str(RW/Path(r['checker']).relative_to(W))]=sha(r['checker'])
 n=proofs[tuple(k)];assert n['window']==[c['started_at'],c['finished_at']];raw={json.loads(l)['text'] for l in (Path(r['checker']).parent/'native.jsonl').read_text().splitlines() if l.strip()}
 for e in n['events']:
  assert c['started_at']<=e['timestamp']<=c['finished_at'];events+=1
  if e['text'] not in raw:missing.append(e)
 assert any('StreamLaunchKernelV1:' in e['text'] for e in n['events'])
 rows.append({'key':k,'checker':r['checker'],'guard':r['guard'],'full_outputs':v,'guard_returncode':0})
assert len(rows)==10 and {tuple(r['key']) for r in rows}=={tuple(k) for k in reg['exact_keys']}
# Same exact authorized diagnostic override as previously accepted14.
F=O/'interim14';orig=(F/'numerics.original.py').read_text();priv=(F/'numerics_private.py').read_text();assert orig.replace('a, b = a.to(dtype), b.to(dtype)','a, b = a.to(device="cpu", dtype=dtype), b.to(device="cpu", dtype=dtype)')==priv
code=(P/'constructor_check.py').read_text().replace("params=json.loads(row['source_reason'].split('本项参数=')[1])","params=PARAMS[row['case_id']]").replace("(P/'CONSTRUCTOR_CHECK.json').write_text(json.dumps(result,indent=2));",'')
pre='import pathlib,hashlib\nPARAMS='+repr(params)+'\nfor p,h in '+repr(remote)+'.items():\n assert hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()==h,p\n'
pre+='rawcache={}\nfor e in '+repr(missing)+':\n if e["file"] not in rawcache:rawcache[e["file"]]=pathlib.Path(e["file"]).read_text(errors="replace").splitlines()\n assert rawcache[e["file"]][e["line"]-1].rstrip()==e["text"].rstrip()\n'
pre+='assert pathlib.Path('+repr(reg['shared_init'])+').read_bytes().startswith('+repr((P/'provider_init.before.py').read_bytes())+')\n'
x=subprocess.run([str(W/'.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'),'openlibing-DevEnv-055409','python3 -S -'],input=pre+code,text=True,capture_output=True,timeout=50);assert x.returncode==0,x.stderr;cpu=json.loads(x.stdout);assert len(cpu['rows'])==10;put(P/'PARENT_CPU_REGISTRY_CHECK.json',cpu)
now=datetime.datetime.now(datetime.timezone.utc).isoformat();review={'at':now,'qualified_rows':rows,'frozen_evidence_hashes_verified':len(manifest),'remote_hashes_verified':len(remote),'native_events_verified':events,'original_boundary':'Byte-identical dsa_adapter to acceptedinterim14, whole originaldecode_mla, selected causalKV layout and independent64query batching, page128. Originalkernel/compiler/math/tolerance unchanged. Per-case native launch timestamps matched fullchecker intervals.','diagnostic_qualification':'Only max_abs error diagnostic CPUconversion as acceptedinterim14; originalreference andvalidator math unchanged. Publicnumerics unchanged.','registration':'Fresh actualdiskregistry constructors with explicitdevice-runtime stubs, originalmodule cachepoison ignored, canonicalreceipt params used, priorprovider prefix preserved. NoNPUreexecution.','new_npu_runs':0};rp=P/'PARENT_REVIEW.json'
with open(str(D/'ACCEPTED.json')+'.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);a=read(D/'ACCEPTED.json');by={tuple(r['key']):r for r in a['rows']};old=len(by);shutil.copy2(D/'ACCEPTED.json',D/('ACCEPTED.before_sgl_attention10_'+now.replace(':','-')+'.json'))
 for r in rows:by.setdefault(tuple(r['key']),dict(r,at=now,accepted=True,host='server2',parent_review=str(rp),qualification=review['diagnostic_qualification'],original_checker_status='verified'))
 a.update(rows=list(by.values()),updated_at=now);put(D/'ACCEPTED.json',a)
review.update(accepted_added=len(by)-old,accepted_total=len(by));put(rp,review);print(json.dumps({'accepted_added':len(by)-old,'accepted_total':len(by),'frozen_hashes':len(manifest),'remote_hashes':len(remote),'native_events':events}))
