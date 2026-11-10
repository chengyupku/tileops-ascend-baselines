import json,hashlib,pathlib,re,shutil,datetime,difflib
W=pathlib.Path('/opt/tileops-all-pairs/workspace'); P=W/'reports/baseline-integration-700-20261026/sgl_gdn_remote38'; Q=W/'reports/baseline-integration-700-20261026/sgl_gdn_register30'; Q.mkdir(exist_ok=True)
snapshot=json.loads((P/'DELIVERY_INTERIM.json').read_text()); scope=[r for r in snapshot['rows'] if r['status']=='verified_unregistered']; assert len(scope)==30
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def rec(p):return {'path':str(p),'sha256':sha(p)}
frozen={}
for n,h in snapshot['frozen_files'].items():
 p=Q/n; assert sha(p)==h,(n,sha(p),h); frozen[n]=rec(p)
rows=[]; deps={}; history={}
for r in scope:
 c=P/r['checker_receipt']; d=json.loads(c.read_text()); g=P/r['guard_receipt']; gd=json.loads(g.read_text()); o=c.parent/'output'; b=g.parent
 assert d['status']=='verified' and d['verification']['checked_outputs']==d['verification']['total_outputs']==2
 assert d['inputs_preserved'] and not d['oracle_source_changed'] and not d['provider_source_changed'];assert gd['exit_code']==0
 assert d['case_snapshot']['declared_output_names']==['o','final_state']; assert not d['case_snapshot']['parameters'].get('use_qk_l2norm_in_kernel',False)
 binding=[(k,v) for k,v in d['source_sha256'].items() if k.endswith('integration700_sgl_gdn_remote38.py')];assert len(binding)==1 and binding[0][1]==snapshot['frozen_files']['frozen_verified/integration700_sgl_gdn_remote38.py']
 loaded=json.loads((o/'LOADED_MODULES.json').read_text()); assert loaded['workloads.numerics']['sha256']==d['oracle_source_sha256']['TileOPs/workloads/numerics.py']
 route=json.loads((o/'ROUTE.json').read_text());assert route['key']==r['key']
 for k,v in {**d['source_sha256'],**d['oracle_source_sha256'],**d.get('reference_source_sha256',{})}.items():
  if 'integration700_sgl_gdn_remote38.py' in k:continue
  p=pathlib.Path(k);p=p if p.is_absolute() else W/p;assert sha(p)==v,(p,sha(p),v);deps[str(p)]=v
 for mod,v in loaded.items():
  if v.get('path') and v.get('sha256'):assert sha(pathlib.Path(v['path']))==v['sha256'];deps[v['path']]=v['sha256']
 native=o/'native.jsonl';counts={};samples={}
 for ln in native.open():
  event=json.loads(ln);t=event['text'];m=re.search(r'StreamLaunchKernel\w*:.*kernel_name=([^,]+)',t)
  if m:
   k=m.group(1);counts[k]=counts.get(k,0)+1
   if k not in samples:samples[k]=event
 native_receipt=rec(native)
 if not counts:
  native=P/g.parent.name/'RETAINED_NATIVE.jsonl'
  lo=datetime.datetime.fromisoformat(d['started_at']);hi=datetime.datetime.fromisoformat(d['finished_at'])
  for ln in native.open():
   event=json.loads(ln);t=event['text'];m=re.search(r'(2026-10-26)-(\d\d:\d\d:\d\d)\.(\d+)\.(\d+)',t); k=re.search(r'StreamLaunchKernel\w*:.*kernel_name=([^,]+)',t)
   if m and k:
    stamp=datetime.datetime.fromisoformat(m[1]+'T'+m[2]+'.'+m[3]+m[4]+'+00:00')
    if lo<=stamp<=hi:
     name=k[1];counts[name]=counts.get(name,0)+1
     if name not in samples:samples[name]=event
  native_receipt=rec(native)
 assert counts,('missing native launch',r['key'])
 maps=(o/'maps.txt').read_text();libs=sorted({l.split()[-1] for l in maps.splitlines() if 'libsgl_kernel_npu.so' in l});assert len(libs)==1
 lib=pathlib.Path(libs[0]);deps[str(lib)]=sha(lib)
 history[b.name]={'guard_log':rec(b/'guard.log'),'historical_runner_snapshot_present':(b/'run_case.py').exists(),'runner_supplied_frozen':frozen['frozen_verified/run_case.original_numerics.frozen.py'],'binding_equality':'exact tested source_sha256 equals frozen bytes; no branch equivalence needed'}
 assert "CASE_DONE " + str([r['operator'],r['case_id']])+" verified" in (b/'guard.log').read_text()
 assert sha(W/'TileOPs/scripts/ascend_pair_check.py')==d['verifier_sha256']
 assert sha(W/'TileOPs/benchmarks/ascend_baselines.py')==d['adapter_sha256']
 assert sha(W/'TileOPs/workloads/sequence_metadata.py')==d['sequence_metadata_sha256']
 rows.append({'key':r['key'],'checker':rec(c),'guard':rec(g),'guard_exit_code':gd['exit_code'],'guard_command':gd['command'],'snapshot':d['case_snapshot'],'seed':d['input_seed'],'verification':d['verification'],'oracle_hashes':d['oracle_source_sha256'],'verifier_sha256':d['verifier_sha256'],'adapter_sha256':d['adapter_sha256'],'sequence_metadata_sha256':d['sequence_metadata_sha256'],'tested_binding':binding[0],'loaded':loaded,'route':route,'evidence':{n:rec(o/n) for n in ['LOADED_MODULES.json','maps.txt','native.jsonl','native.summary.json','ROUTE.json','ENVIRONMENT.json','GUARD_RUNNING.json']},'loaded_sgl_library':rec(lib),'attributed_native_receipt':native_receipt,'native_launch_counts':counts,'native_launch_samples':list(samples.values())[:8]})
for rel in ['TileOPs/scripts/ascend_pair_check.py','TileOPs/benchmarks/ascend_all_pairs/registry.py','TileOPs/benchmarks/ascend_baselines.py','TileOPs/benchmarks/ascend_math_more.py','TileOPs/benchmarks/ascend_delta_reference.py','TileOPs/workloads/sequence_metadata.py','tools/baseline_protocol.py','TileOPs/workloads/linear_attention/input_values.py','TileOPs/workloads/device.py','TileOPs/benchmarks/ascend_all_pairs/sgl/__init__.py']:
 p=W/rel
 if p.is_file():deps[str(p)]=sha(p);dst=Q/'source_review'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
for path in ['/opt/tileops-all-pairs/bin/allpairs-python','/opt/tileops-all-pairs/control/private_python.py']:
 p=pathlib.Path(path);deps[path]=sha(p);shutil.copyfile(p,Q/p.name)
result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope_count':len(rows),'frozen':frozen,'history':history,'dependencies':deps,'rows':rows,'native_ownership':'Original SGL complete chunk_gated_delta_rule_native calls upstream torch operations, dispatched to CANN. Not SGL custom kernel launch; no substituted implementation.'}
(Q/'AUDIT.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result))
