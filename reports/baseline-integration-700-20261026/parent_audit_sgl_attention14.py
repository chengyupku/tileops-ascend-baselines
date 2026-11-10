from pathlib import Path
import json,hashlib,datetime,subprocess
W=Path('/root/dyq/workspace');D=W/'reports/baseline-integration-700-20261026';O=D/'sgl_attention_recovery32';F=O/'interim14';R=D/'sgl_attention_register14';RW=Path('/opt/tileops-all-pairs/workspace');RO=RW/O.relative_to(W);RR=RW/R.relative_to(W)
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows=read(R/'INPUT_INTERIM_SNAPSHOT.json')['rows'];assert len(rows)==14
manifest=read(F/'EVIDENCE_HASHES.json')
for p,h in manifest.items():assert sha(O/p)==h,p
orig=(F/'numerics.original.py').read_text();priv=(F/'numerics_private.py').read_text();old='a, b = a.to(dtype), b.to(dtype)';new='a, b = a.to(device="cpu", dtype=dtype), b.to(device="cpu", dtype=dtype)';assert orig.replace(old,new)==priv and orig.count(old)==1
remote={};stage=read(R/'stage_hashes.json')
for p,h in stage.items():
 remote[str(RR/p)]=h
 if (F/p).exists() and p!='provider_registration.append.py':assert sha(F/p)==h,p
for p,h in read(F/'CONTRACT_MATCH.json').items():assert sha(W/p)==h['local']==h['remote'];remote[str(RW/p)]=h['remote']
for r in read(F/'SOURCE_HASHES.json'):
 remote[r['path']]=r['sha256'];remote[r['resolved']]=r['sha256']
proofs={tuple(x['key'][k] for k in ('baseline','operator','case_id')):x for x in read(F/'NATIVE_PROOF.json')['rows']};out=[];events=0;missing_events=[]
for r in rows:
 key=tuple(r[k] for k in ('baseline','operator','case_id'));c=read(O/r['checker_receipt']);g=read(O/r['guard_receipt']);v=c['verification'];assert c['status']=='verified' and v['checked_outputs']==v['total_outputs']>0 and not v['unchecked_reason'];assert c['inputs_preserved'] and not c['oracle_source_changed'] and not any(c.get(k) for k in ('sync_error','cleanup_error','error'))
 assert c['input_seed']==int.from_bytes(hashlib.sha256((key[1]+'::'+key[2]).encode()).digest()[:4],'big');assert c['verifier_sha256']==sha(W/'TileOPs/scripts/ascend_pair_check.py');assert g['exit_code'] in (0,1)
 if g['exit_code']==1:assert r['guard_receipt']=='attempts/reps03/GUARD_RESULT.json' or g['batch_guard_receipt']=='batches/reps03/GUARD_RESULT.json';assert key[1] in ('DSADecodeWithKVCacheFwdOp','GQAVarlenFwdOp');assert "KeyError: 'source_sha256'" in (O/'batches/reps03/guard.log').read_text()
 for p,h in c['source_sha256'].items():
  pp=Path(p)
  if str(RO) in p and '/libraries/' not in p:
   assert sha(F/pp.name)==h,(p,h);remote[str(RR/pp.name) if pp.name in stage else p]=h
  elif '/libraries/' in p:remote[str(pp if pp.is_absolute() else RW/pp)]=h
 for p,h in c['oracle_source_sha256'].items():
  pp=Path(p);pp=pp if pp.is_absolute() else RW/pp;remote[str(pp)]=h
 n=proofs[key];assert n['window']==[c['started_at'],c['finished_at']];raw={json.loads(line)['text'] for line in (O/Path(r['checker_receipt']).parent/'native.jsonl').read_text().splitlines() if line.strip()}
 for e in n['events']:
  if e['text'] not in raw:missing_events.append(e)
  assert c['started_at']<=e['timestamp']<=c['finished_at'];events+=1
 assert n['native_launch_count']>0 and any('StreamLaunchKernelV1:' in e['text'] for e in n['events'])
 for f in ('checker_receipt',):remote[str(RO/r[f])]=sha(O/r[f])
 out.append({'key':list(key),'checker':str(O/r['checker_receipt']),'guard':str(O/r['guard_receipt']),'full_outputs':v,'guard_returncode':g['exit_code'],'native_kernels':n['kernels']})
code='''import sys, pathlib, hashlib, json, types, importlib, importlib.abc
sys.dont_write_bytecode=True
rawlog_cache={}
for e in MISSING:
 lines=rawlog_cache.setdefault(e['file'],pathlib.Path(e['file']).read_text(errors='replace').splitlines())
 assert lines[e['line']-1]==e['text'],e
for p,h in HASHES.items():assert hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()==h,p
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch','torch_npu','triton','sgl_kernel_npu'):raise RuntimeError('Device import forbidden '+name)
sys.meta_path.insert(0,Block());W=pathlib.Path('/opt/tileops-all-pairs/workspace');R=pathlib.Path(REGROOT);sys.path[:0]=[str(W/'TileOPs'),str(W/'TileOPs/src'),str(W),str(R)]
from benchmarks.ascend_all_pairs import registry,sgl
m=importlib.import_module('integration700_sgl_attention_register14')
class Reached(Exception):pass
records=[]
for key in KEYS:
 op,cid=key[1:];fn=m.PROVIDERS[op]
 def trace(frame,event,arg):
  if event=='call' and frame.f_code is fn.__code__:raise Reached()
 case=types.SimpleNamespace(id=cid,_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=op)))
 sys.setprofile(trace)
 try:registry.candidate_factory(case,'sgl-kernel-npu')()
 except Reached:pass
 else:raise AssertionError('Factory not reached')
 finally:sys.setprofile(None)
 records.append({'key':key,'file':fn.__code__.co_filename,'function':fn.__name__})
for op in sgl._ar14_keys:
 token=object();assert sgl._ar14_route(op,frozenset(),lambda c:token)(types.SimpleNamespace(id='outside_scope')) is token
append=(R/'provider_registration.append.py').read_text();assert append in pathlib.Path(sgl.__file__).read_text()
assert not any(n in sys.modules for n in ('torch','torch_npu','triton','sgl_kernel_npu'))
print(json.dumps({'hashes_verified':len(HASHES),'exact_factory_entry_routes':records,'fallbacks':len(sgl._ar14_keys),'device_imports':False,'npu_runs':0,'note':'Actual factory entry intercepted before body to prohibit device imports; numerical body evidenced by original canonical receipts.'}))
'''
code='MISSING='+repr(missing_events)+'\nHASHES='+repr(remote)+'\nREGROOT='+repr(str(RR))+'\nKEYS='+repr([x['key'] for x in out])+'\n'+code
p=subprocess.run([str(W/'.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'),'openlibing-DevEnv-055409','python3 -S -'],input=code,text=True,capture_output=True,timeout=50);assert p.returncode==0,p.stderr
live=json.loads(p.stdout);(D/'PARENT_SGL_ATTENTION14_LIVE_CHECK.json').write_text(json.dumps(live,indent=2))
review={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'qualified_rows':out,'local_frozen_evidence_hashes':len(manifest),'remote_hashes_verified':len(remote),'native_log_events_checked':events,'original_boundary':'Full original SGL attention/cache-write/RoPE APIs. Adapters pack/page/slice causal intervals and independent heads/sequences; DSA gathers selected causal KV. No changed kernel, compiler, math or tolerance. Native named launch events retained inside each checker interval.','diagnostic_qualification':'Private numerics max_abs error-reporting conversion moved to CPU only; exact single replacement verified. Actual validator/reference arithmetic unchanged, public numerics unchanged. Full checker rerun requires this disclosed diagnostic setup until shared reporting issue is resolved.','guard_qualification':'DSA512 and uniform varlen successes completed full checker+sync before subsequent third case build failure/runner KeyError aborted shared batch with guard1. Preserve guard1, do not relabel CLI success.','bytecode_note':read(F/'BYTECODE_NOTE.json'),'new_npu_runs':0,'live_registry_proof':str(D/'PARENT_SGL_ATTENTION14_LIVE_CHECK.json')}
(D/'PARENT_SGL_ATTENTION14_REVIEW.json').write_text(json.dumps(review,ensure_ascii=False,indent=2));print(json.dumps({'qualified':len(out),'local_hashes':len(manifest),'remote_hashes':len(remote),'events':events}))
