import pathlib,json,hashlib,fcntl,sys,types,os,datetime,shutil
sys.dont_write_bytecode=True
D=pathlib.Path(__file__).resolve().parent;W=D.parents[1];P=D/'nn_registration02';rows=json.loads((P/'REGISTRATION.json').read_text())['rows'];assert len(rows)==7
cache={};native_refs={}
def sha(p):
 p=pathlib.Path(p);p=p if p.is_absolute() else W/p
 if str(p) not in cache:cache[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 return cache[str(p)]
def refs(x):
 if isinstance(x,dict):
  if 'path' in x and 'sha256' in x:assert sha(x['path'])==x['sha256'],x['path']
  for v in x.values():refs(v)
 elif isinstance(x,list):
  for v in x:refs(v)
for r in rows:
 refs(r)
 c=json.loads(pathlib.Path(r['checker']['path']).read_text());v=c['verification']
 assert c['status']=='verified' and c['inputs_preserved'] and not c['oracle_source_changed'] and v['checked_outputs']==v['total_outputs']>0
 assert c['input_seed']==r['input_seed']==int.from_bytes(hashlib.sha256((r['key'][1]+'::'+r['key'][2]).encode()).digest()[:4],'big')
 g=json.loads(pathlib.Path(r['guard']['path']).read_text());assert g['returncode']==0
 gm=json.loads(pathlib.Path(r['guard_metadata']['path']).read_text());assert gm['owner']=='baseline700-20261026/nn_recovery02' and gm['acquired_at']<c['started_at']
 assert sha(r['actual_tested_factory'].removesuffix(':make'))==r['adapter_sha256']==sha(r['module'])==r['module_sha256']
 assert sha(r['route_module'])==r['route_module_sha256']
 for mapping in ['source_runtime_hashes_checked','oracle_source_sha256']:
  for p,h in r[mapping].items():assert sha(p)==h,p
 assert r['owned_events']
 for e in r['owned_events']:
  assert c['started_at']<=e['at']<=c['finished_at']
  assert e['kernel'] in e['compile']['text'] and e['kernel'] in e['launch']['text'] and 'classicpilot_nn_impl.dynamic.mat_mul_v3' in e['compile']['text']
  for typ in ['launch','compile']:
   event=e[typ];native_refs.setdefault(event['file'],{})[event['line']]=event['text']
for p,lines in native_refs.items():
 with open(p,errors='replace') as f:
  matched=0
  for i,line in enumerate(f,1):
   if i in lines:assert line.rstrip('\n')==lines[i];matched+=1
 assert matched==len(lines)
source_compare=json.loads((D/'nn_recovery02/SOURCE_COMPARE.json').read_text())
for r in source_compare:assert r['identical'] and sha(r['source'])==sha(r['build'])==r['sha256']
sys.path.insert(0,str(W/'TileOPs'))
with open('/tmp/baseline700-provider-ops_nn.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 from benchmarks.ascend_all_pairs import ops_nn,registry
 m=ops_nn._nn_registration02
 assert sha(m.__file__)==rows[0]['route_module_sha256']
 assert m._load().__file__==rows[0]['module']
 before=next(P.glob('provider_before_*.py')).read_text();current=(W/'TileOPs/benchmarks/ascend_all_pairs/ops_nn/__init__.py').read_text();assert current.startswith(before)
 original=m._load;m._load=lambda:types.SimpleNamespace(make=lambda c:('expected',c.id))
 os.environ['OPS_NN_VENDOR']=str(m._VENDOR);os.environ['ASCEND_CUSTOM_OPP_PATH']=str(m._VENDOR)
 for r in rows:
  op,cid=r['key'][1:];case=types.SimpleNamespace(id=cid,_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=op)))
  assert registry.candidate_factory(case,'ops-nn')()==('expected',cid)
 case.id='__unknown__';assert ops_nn._nn_registration02_route(case,_prior=lambda c:'previous')=='previous'
 m._load=original
 assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
now=datetime.datetime.now(datetime.timezone.utc).isoformat();review_path=D/'PARENT_NN_REGISTRATION02_REVIEW.json';out=D/'ACCEPTED.json'
with open(str(out)+'.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);ledger=json.loads(out.read_text());by={tuple(r['key']):r for r in ledger['rows']};before=len(by)
 for r in rows:by[tuple(r['key'])]=dict(r,accepted=True,at=now,parent_review=str(review_path))
 ledger.update(rows=list(by.values()),updated_at=now);tmp=out.with_suffix('.parent.tmp');tmp.write_text(json.dumps(ledger,ensure_ascii=False,indent=2));tmp.replace(out)
review={'at':now,'accepted_added':len(by)-before,'accepted_total':len(by),'keys':[r['key'] for r in rows],'hashes_checked':len(cache),'original_build_file_pairs_checked':len(source_compare),'native_logs_raw_lines_checked':sum(len(v) for v in native_refs.values()),'live_registry_exact_keys':7,'device_runs':0,'numeric_reruns':0,'boundary_review':'Complete original MatMulV3; lossless BF16/FP16->FP32 input binding, canonical transpose/contiguous and final output cast only; no new arithmetic or kernel changes. Full canonical independent reference and all outputs checked. Runtime must use actual recorded classicpilot_nn vendor; wrapper enforces it.','formal_module':rows[0]['route_module'],'formal_module_sha256':rows[0]['route_module_sha256'],'frozen_adapter_sha256':rows[0]['module_sha256']}
review_path.write_text(json.dumps(review,ensure_ascii=False,indent=2))
for folder in ['nn_recovery02','nn_registration02']:
 (D/folder/'PARENT_REVIEW.json').write_text(json.dumps({'at':now,'accepted':7,'review':str(review_path),'remaining_unaccepted':16 if folder=='nn_recovery02' else 0,'remaining_scope':str(D/'NN_RECOVERY02_ROOT_QUEUED16.json') if folder=='nn_recovery02' else None},ensure_ascii=False,indent=2))
state_path=D/'STATE.json';state=json.loads(state_path.read_text());state.update(completed=len(by),parent_accepted_completed=len(by),last_status_accepted=len(by),last_acceptance_review=str(review_path),last_status_at=now);tmp=state_path.with_suffix('.parent.tmp');tmp.write_text(json.dumps(state,ensure_ascii=False,indent=2));tmp.replace(state_path)
mem=W/'.agent_memory/THREADS/baseline-inspector-20261023/STATE.json'
with open(str(mem)+'.lock','a') as a,open(mem.parent/'STATE.lock','a') as b:
 fcntl.flock(a,fcntl.LOCK_EX);fcntl.flock(b,fcntl.LOCK_EX);shutil.copy2(mem,mem.with_name('STATE.before_nn7_accept_'+datetime.datetime.now().strftime('%Y%m%dT%H%M%S')+'.json'))
 memory=json.loads(mem.read_text());memory['integration_700_20261026'].update(parent_accepted_completed=len(by),last_acceptance_review=str(review_path),last_status_at=now);tmp=mem.with_suffix('.parent.tmp');tmp.write_text(json.dumps(memory,ensure_ascii=False,indent=2));tmp.replace(mem)
print(json.dumps(review,ensure_ascii=False))
