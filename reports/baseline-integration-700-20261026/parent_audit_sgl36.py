import json,pathlib,hashlib,subprocess,fcntl,datetime,shutil
D=pathlib.Path(__file__).resolve().parent;W=D.parents[1];P=D/'sgl_linalg_register36';O=D/'sgl_linalg36';RW='/opt/tileops-all-pairs/workspace'
e=json.loads((P/'EVIDENCE.json').read_text());reg=json.loads((P/'REGISTRATION.json').read_text());native=json.loads((P/'NATIVE_AUDIT.json').read_text());rows=e['rows'];assert len(rows)==reg['registered_count']==36
cache={};remote_hashes={};canonical_sources=0
def sha(p):
 p=pathlib.Path(p)
 if str(p) not in cache:cache[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 return cache[str(p)]
def mirror(p):return pathlib.Path(str(p).replace(RW,str(W),1))
for r in rows:
 for name in ['checker','guard','route']:assert sha(mirror(r[name]['path']))==r[name]['sha256'],r[name]['path']
 c=json.loads(mirror(r['checker']['path']).read_text());g=json.loads(mirror(r['guard']['path']).read_text());v=c['verification']
 assert c['status']=='verified' and c['inputs_preserved'] and not c['provider_source_changed'] and not c['oracle_source_changed'];assert v['checked_outputs']==v['total_outputs']==1
 assert [c['library'],c['op'],c['case_id']]==r['key'] and c['input_seed']==r['seed']==int.from_bytes(hashlib.sha256((c['op']+'::'+c['case_id']).encode()).digest()[:4],'big')
 assert g['exit_code']==0 and g['command'][g['command'].index('--owner')+1]=='baseline700-20261026/sgl_linalg36'
 meta=json.loads((mirror(r['checker']['path']).parent/'acquired.json').read_text())['metadata'];assert meta['owner']=='baseline700-20261026/sgl_linalg36' and meta['phase']=='running'
 assert r['route']['adapter_hash']==reg['module_sha256']==e['adapter_sha256']
 for path,h in c['source_sha256'].items():remote_hashes[path if path.startswith('/') else RW+'/'+path]=h
 for path,h in c['oracle_source_sha256'].items():
  assert sha(W/path)==h,path;remote_hashes[RW+'/'+path]=h
for path,v in e['hashes'].items():assert v['actual']==v['expected'];remote_hashes[path]=v['expected']
for path,h in native['files'].items():remote_hashes[path]=h
source=e['source_directory'].rstrip('/')
for path,h in remote_hashes.items():
 if path.startswith(source+'/csrc/'):
  original=W/'ascend_baselines/sgl-kernel-npu'/path[len(source)+1:]
  assert sha(original)==h,str(original);canonical_sources+=1
assert canonical_sources>10
remote_hashes[reg['module']]=reg['module_sha256'];assert sha(O/'integration700_sgl_linalg36.py')==reg['module_sha256']
code='''import sys,pathlib,hashlib,json,fcntl,types
sys.dont_write_bytecode=True;sys.path.insert(0,'/opt/tileops-all-pairs/workspace/TileOPs')
hashes=HASHES
keys=KEYS
out=[]
for p,h in hashes.items():assert hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()==h,p
with open('/tmp/baseline700-provider-sgl.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 from benchmarks.ascend_all_pairs import sgl,registry
 from benchmarks.ascend_all_pairs.sgl import integration700_sgl_linalg_register36 as m
 original=m.build;aliases={k:v for k,v in vars(sgl).items() if v is original}
 assert aliases
 stub=lambda c:('expected',c._call.signature.name,c.id)
 m.build=stub
 for k in aliases:setattr(sgl,k,stub)
 for key in keys:
  c=types.SimpleNamespace(id=key[2],_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=key[1])))
  assert registry.candidate_factory(c,key[0])()==('expected',key[1],key[2]);out.append(key)
 m.build=original
 for k,v in aliases.items():setattr(sgl,k,v)
 provider=pathlib.Path(sgl.__file__).read_bytes();backup=pathlib.Path('/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/sgl_linalg_register36/provider_init.before.py').read_bytes();assert provider.startswith(backup)
 assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
 print(json.dumps({'checked_keys':out,'critical_hashes':len(hashes),'provider_prefix_preserved':True,'device_imported':False,'formal_module_sha256':hashlib.sha256(pathlib.Path(m.__file__).read_bytes()).hexdigest()}))
'''.replace('hashes=HASHES','hashes='+repr(remote_hashes)).replace('keys=KEYS','keys='+repr([r['key'] for r in rows]))
p=subprocess.run(['/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh','openlibing-DevEnv-055409','python3 -S -'],input=code,text=True,capture_output=True,timeout=50);assert p.returncode==0,p.stderr
proof=json.loads(p.stdout);(D/'PARENT_SGL36_LIVE_ROUTE_CHECK.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2))
now=datetime.datetime.now(datetime.timezone.utc).isoformat();review_path=D/'PARENT_SGL36_REVIEW.json';out=D/'ACCEPTED.json';registration={tuple(r['key']):r for r in reg['rows']}
with open(str(out)+'.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);ledger=json.loads(out.read_text());by={tuple(r['key']):r for r in ledger['rows']};before=len(by)
 for r in rows:
  row={k:v for k,v in r.items() if k not in ['case_snapshot','observed_calls']};row.update(accepted=True,at=now,parent_review=str(review_path),formal_registration=registration[tuple(r['key'])],runtime_recipe=str(O/'DELIVERY.json'));by[tuple(r['key'])]=row
 ledger.update(rows=list(by.values()),updated_at=now);tmp=out.with_suffix('.parent.tmp');tmp.write_text(json.dumps(ledger,ensure_ascii=False,indent=2));tmp.replace(out)
review={'at':now,'accepted_added':len(by)-before,'accepted_total':len(by),'scope':36,'full_outputs_checked':36,'live_registry_exact_keys':36,'local_receipt_hashes':len(cache),'remote_source_runtime_hashes':len(remote_hashes),'original_source_files_matched':canonical_sources,'device_runs':0,'numeric_reruns':0,'module':reg['module'],'module_sha256':reg['module_sha256'],'boundary_review':'Byte-identical tested adapter formally registered; original sgemmv_shrink fullK, scale1, independent columns/group packing, zero padding/transpose/output cast. No splitK math or internal schedule changes. Empty M groups return empty output only. Actual run_case injects provider before unchanged fullchecker; run_batch synchronizes each case; clean enclosingguard0. Original sourceCpp matches localbaseline, loaded extension ownership/source binding and symbol inspected.','route_evidence':str(D/'PARENT_SGL36_LIVE_ROUTE_CHECK.json'),'limitations':'Recorded15.6 seconds is submission-to-child-start including startup, not isolated lock queue; successful-case reruns0, two prior dependency failures still counted as waste.'}
review_path.write_text(json.dumps(review,ensure_ascii=False,indent=2));(O/'PARENT_REVIEW.json').write_text(json.dumps(review,ensure_ascii=False,indent=2));(P/'PARENT_REVIEW.json').write_text(json.dumps({'at':now,'accepted':36,'review':str(review_path)},ensure_ascii=False,indent=2))
state_path=D/'STATE.json';state=json.loads(state_path.read_text());state.update(completed=len(by),parent_accepted_completed=len(by),last_status_accepted=len(by),last_status_at=now,last_acceptance_review=str(review_path));tmp=state_path.with_suffix('.parent.tmp');tmp.write_text(json.dumps(state,ensure_ascii=False,indent=2));tmp.replace(state_path)
mem=W/'.agent_memory/THREADS/baseline-inspector-20261023/STATE.json'
with open(str(mem)+'.lock','a') as a,open(mem.parent/'STATE.lock','a') as b:
 fcntl.flock(a,fcntl.LOCK_EX);fcntl.flock(b,fcntl.LOCK_EX);shutil.copy2(mem,mem.with_name('STATE.before_sgl36_'+datetime.datetime.now().strftime('%Y%m%dT%H%M%S')+'.json'));m=json.loads(mem.read_text());m['integration_700_20261026'].update(parent_accepted_completed=len(by),last_acceptance_review=str(review_path),last_status_at=now);tmp=mem.with_suffix('.parent.tmp');tmp.write_text(json.dumps(m,ensure_ascii=False,indent=2));tmp.replace(mem)
print(json.dumps(review,ensure_ascii=False))
