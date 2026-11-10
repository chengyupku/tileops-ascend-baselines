from pathlib import Path
import json,hashlib,subprocess,datetime,fcntl,os,shutil,sys,yaml
D=Path(__file__).resolve().parent;W=D.parents[1];R=D/'audit_attention15';rows=json.loads((R/'RECOMMENDED_ACCEPT.json').read_text())['rows'];assert len(rows)==15
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,j):
 p=Path(p);t=p.with_suffix('.tmp-parent');t.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n');os.replace(t,p)
local={};remote={};native=[];timeouts=[]
def checkrefs(obj):
 if isinstance(obj,dict):
  p=obj.get('path');h=obj.get('sha256')
  if p and h:
   if p.startswith('/root/dyq/workspace/'):
    assert sha(p)==h,p;local[p]=h
    if obj.get('remote_path'):remote[obj['remote_path']]=h
   elif p.startswith('/opt/tileops-all-pairs/'):remote[p]=h
  for v in obj.values():checkrefs(v)
 elif isinstance(obj,list):
  for v in obj:checkrefs(v)
manifest=yaml.safe_load((W/'TileOPs/src/tileops/manifest/spec/attention.yaml').read_text())
for r in rows:
 assert r['accept_recommended'] and not r['issues'];checkrefs(r)
 c=read(r['checker']['path']);g=read(r['guard']['path']);v=c['verification'];key=r['key']
 assert [c['library'],c['op'],c['case_id']]==key
 assert c['status']=='verified' and c['inputs_preserved'] and not c['provider_source_changed'] and not c['oracle_source_changed']
 assert v['checked_outputs']==v['total_outputs']==1 and v['unchecked_reason'] is None
 assert c['input_seed']==r['seed']==int.from_bytes(hashlib.sha256((key[1]+'::'+key[2]).encode()).digest()[:4],'big')
 assert c['case_snapshot']==r['canonical']['snapshot']
 workload=next(x for x in manifest[key[1]]['workloads'] if key[2].startswith(x['label']+'-'))
 assert workload==r['canonical']['workload'];assert r['canonical']['full_declared_shapes_dtypes_parameters_match'] and r['canonical']['observed_input_output_shapes_dtypes_match']
 ref=c['observed_calls']['reference'][0];cand=c['observed_calls']['candidate'][0];assert ref['inputs']==cand['inputs'] and ref['returned'] and cand['returned']
 snap=c['case_snapshot'];decl=snap['declared_tensors']
 for name,got in zip(snap['declared_input_names'],cand['inputs']['items'],strict=True):
  assert got['shape']==decl[name][0] and got['dtype']==decl[name][1],(key,name)
 assert cand['outputs']['shape']==decl[snap['declared_output_names'][0]][0] and cand['outputs']['dtype']==decl[snap['declared_output_names'][0]][1]
 assert r['tested_registered_identity']['tested']['sha256']==r['formal_module']['sha256']
 assert g['returncode']==r['guard']['returncode'] in [0,124]
 end=datetime.datetime.fromisoformat(c['finished_at']).timestamp();assert g['end']>end
 assert r['completion']['explicit_sync_and_cleanup_before_receipt']
 start=read(r['completion']['guard_running']['path']);assert start['guard']['phase']=='running' and start['guard']['owner']=='baseline700-20261026/transformer_attention_misc'
 if g['returncode']==124:
  log=Path(r['guard']['log']['path']).read_text(errors='replace');assert c['case_id'] in log and 'verified' in log
  assert 'tbe/common/repository_manager' in log
  timeouts.append({'key':key,'guard_returncode':124,'completed_before_end_seconds':g['end']-end,'checker_status':'verified','sync_cleanup_before_receipt':True})
 for name in ['native_launch','native_source_ownership']:
  assert r['source_provenance'][name]
  native.extend(r['source_provenance'][name])
 for path,h in c['oracle_source_sha256'].items():assert sha(W/path)==h
critical=read(R/'CRITICAL_SOURCE_HASHES.json')
for p in critical:assert p['expected']==p['sha256'];remote[p['path']]=p['sha256']
original={}
for p,h in remote.items():
 marker='/ascend_baselines/ops-transformer/'
 if marker in p:
  rel=p.split(marker,1)[1];localp=W/'ascend_baselines/ops-transformer'/rel
  if localp.is_file():assert sha(localp)==h,p;original[rel]=h
assert len(original)>5
# The unchanged full checker writes its receipt only after synchronization and cleanup.
src=Path(rows[0]['completion']['checker_source']['path']).read_text();assert src.index('torch.npu.synchronize()',src.index('    finally:'))<src.index('args.output.write_text',src.index('    finally:'));assert src.index('candidate.cleanup()',src.index('    finally:'))<src.index('args.output.write_text',src.index('    finally:'))
code='''import pathlib,hashlib,json,sys,importlib,types,fcntl
sys.dont_write_bytecode=True;sys.path.insert(0,'/opt/tileops-all-pairs/workspace/TileOPs')
class Block:
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','torch_npu','acl','ctypes'):raise RuntimeError('forbidden device import '+fullname)
sys.meta_path.insert(0,Block())
hashes=HASHES
for p,h in hashes.items():assert hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()==h,p
native=NATIVE
for e in native:
 content=pathlib.Path(e['path']).read_text(errors='replace')
 assert e['text'] in content,e
 e['parent_verified_line']=content[:content.index(e['text'])].count(chr(10))+1
rows=ROWS
routes=[]
with open('/tmp/baseline700-provider-ops_transformer.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 from benchmarks.ascend_all_pairs import ops_transformer as p,registry
 for r in rows:
  mn,fn=r['formal_factory'].rsplit('.',1);m=importlib.import_module(mn);old=getattr(m,fn);aliases={k:v for k,v in vars(p).items() if v is old}
  stub=lambda *args:('exact',*r['key']);setattr(m,fn,stub)
  for k in aliases:setattr(p,k,stub)
  case=types.SimpleNamespace(id=r['key'][2],_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=r['key'][1])))
  assert registry.candidate_factory(case,r['key'][0])()==stub();routes.append(r['key'])
  setattr(m,fn,old)
  for k,v in aliases.items():setattr(p,k,v)
print(json.dumps({'routes':routes,'hashes':len(hashes),'native_exact_lines':len(native),'native_verified_locations':native,'device_imports':False}))
'''.replace('hashes=HASHES','hashes='+repr(remote)).replace('native=NATIVE','native='+repr(native)).replace('rows=ROWS','rows='+repr([{'key':r['key'],'formal_factory':r['formal_factory']} for r in rows]))
p=subprocess.run(['/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh','openlibing-DevEnv-055409','python3 -S -'],input=code,text=True,capture_output=True,timeout=50);assert p.returncode==0,p.stderr
proof=json.loads(p.stdout);put(D/'PARENT_ATTENTION15_LIVE_CHECK.json',proof)
now=datetime.datetime.now(datetime.timezone.utc).isoformat();rp=D/'PARENT_ATTENTION15_REVIEW.json';ledgerpath=D/'ACCEPTED.json'
review={'at':now,'scope':15,'local_receipt_hashes':len(local),'remote_critical_hashes':len(remote),'original_local_sources_matched':len(original),'native_exact_lines':len(native),'actual_registry_routes':15,'device_runs':0,'full_outputs':15,'guard124_qualifications':timeouts,'boundary':'Unmodified complete original FlashAttentionScore native API. DSA sparse indices/sentinel/compressed causal limits represented as boolean mask metadata. MLA Q/K feature concat and V prefix preserve full attention arithmetic; no new attention algorithm or altered original kernels. Source/build/vendor ownership matched with actual native launches.','live_check':str(D/'PARENT_ATTENTION15_LIVE_CHECK.json'),'guard_note':'Three runtime-success cases retain enclosing CLI guard124; complete checker receipt written after explicit device synchronization and GraphCall cleanup, before Python/TBE teardown timeout. No guard76 receipts accepted.'}
put(rp,review)
with open(str(ledgerpath)+'.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);ledger=read(ledgerpath);by={tuple(x['key']):x for x in ledger['rows']};before=len(by)
 for r in rows:
  if tuple(r['key']) not in by:by[tuple(r['key'])]={**r,'accepted':True,'at':now,'parent_review':str(rp)}
 ledger.update(rows=list(by.values()),updated_at=now);put(ledgerpath,ledger)
review.update(accepted_added=len(by)-before,accepted_total=len(by));put(rp,review);put(R/'PARENT_REVIEW.json',{'at':now,'scope':15,'accepted':15,'review':str(rp)})
s=read(D/'STATE.json');s.update(completed=len(by),parent_accepted_completed=len(by),last_status_accepted=len(by),last_status_at=now,last_acceptance_review=str(rp));put(D/'STATE.json',s)
M=W/'.agent_memory/THREADS/baseline-inspector-20261023/STATE.json'
with open(str(M)+'.lock','a') as a,open(M.parent/'STATE.lock','a') as b:
 fcntl.flock(a,fcntl.LOCK_EX);fcntl.flock(b,fcntl.LOCK_EX);shutil.copy2(M,M.with_name('STATE.before_attention15_'+now.replace(':','-')+'.json'));j=read(M);j['integration_700_20261026'].update(parent_accepted_completed=len(by),last_acceptance_review=str(rp),last_status_at=now);put(M,j)
print(json.dumps({k:review[k] for k in ['accepted_added','accepted_total','local_receipt_hashes','remote_critical_hashes','native_exact_lines','original_local_sources_matched']}))
