import json,hashlib,pathlib,datetime,re
P=pathlib.Path(__file__).resolve().parent; R=P.with_name('sgl_attention_recovery32')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rows=[]; hashes={}; remote={}
for delta in ['delta03','delta07']:
 d=read(R/('DELIVERY_'+delta.upper()+'.json')); h=read(R/delta/'EVIDENCE_HASHES.json')
 for f,v in h.items():assert sha(R/f)==v,f
 hashes[delta]=len(h)
 original=(R/delta/'numerics.original.py').read_text(); private=(R/delta/'numerics_private.py').read_text()
 assert original.replace('a, b = a.to(dtype), b.to(dtype)','a, b = a.to(device="cpu", dtype=dtype), b.to(device="cpu", dtype=dtype)')==private
 native=read(R/delta/'NATIVE_PROOF.json');assert not native['missing_native_launch_keys']
 for s in read(R/delta/'SOURCE_HASHES.json'):remote[s['path']]=s['sha256']
 for row in d['rows']:
  c=read(R/row['checker_receipt']);g=read(R/row['guard_receipt']);run=(R/row['checker_receipt']).parent;key=[row[x] for x in ['baseline','operator','case_id']]
  assert [c['library'],c['op'],c['case_id']]==key==g['key']
  assert c['status']=='verified' and g['exit_code']==0 and c['inputs_preserved'] and not c['oracle_source_changed'] and not c['provider_source_changed']
  assert c['verification']['checked_outputs']==c['verification']['total_outputs']==1 and c['verification']['unchecked_reason'] is None
  assert c['input_seed']==int.from_bytes(hashlib.sha256(f"{c['op']}::{c['case_id']}".encode()).digest()[:4],'big')
  snap=c['case_snapshot'];params=json.loads(row['source_reason'].split('本项参数=')[1]);b,s,sk,h,dim,k,tail=[params[x] for x in ['B','S','S_kv','H','D','K','dim_tail']]
  expected={'q':[[b,s,h,dim+tail],'bfloat16'],'kv':[[b,sk,1,dim+tail],'bfloat16'],'indices':[[b,s,1,k],'int32'],'o':[[b,s,h,dim],'bfloat16']}
  assert snap['declared_tensors']==expected
  assert snap['parameters']['dim_tail']==tail and snap['parameters']['stride_kv']==1 and snap['parameters']['q_start_index_s']==params['q_start_index_s'] and snap['parameters']['sm_scale'] is None
  for which in ['reference','candidate']:
   call=c['observed_calls'][which][0];assert call['returned']
   for name,tensor in zip(['q','kv','indices'],call['inputs']['items']):assert [tensor['shape'],tensor['dtype']]==expected[name]
   assert [call['outputs']['shape'],call['outputs']['dtype']]==expected['o']
  assert c['source_sha256']['reports/baseline-integration-700-20261026/sgl_attention_recovery32/dsa_adapter.py']==sha(R/delta/'dsa_adapter.py')
  assert sha(run/'RUNNER.py')==sha(R/delta/'run_case.py')
  assert sha(run/'ENVIRONMENT.sh')==sha(R/delta/'baseline-run')
  assert c['diagnostic_only_override']['private_sha256']==sha(R/delta/'numerics_private.py')
  route=read(run/'ROUTE.json');assert route['key']==key and route['provider'].endswith('/sgl_attention_recovery32/dsa_adapter.py')
  guard=read(run/'GUARD_RUNNING.json');assert guard['phase']=='running';pid=guard['child']['pid']
  n=next(n for n in native['rows'] if n['key']==row['key']);assert n['window']==[c['started_at'],c['finished_at']] and n['native_launch_count']>0
  events=[json.loads(line) for line in (run/'native.jsonl').read_text().splitlines()];valid=[]
  for ev in n['events']:
   assert any(all(e.get(k)==ev[k] for k in ['file','line','text','captured_at']) for e in events)
   assert f'({pid},python)' in ev['text'] and '/'+run.name+'/' in ev['file']
   assert c['started_at']<=ev['timestamp']<=c['finished_at'];valid.append(ev)
  loaded=read(run/'LOADED_MODULES.json')['modules'];host=loaded['sgl_kernel_npu.attention.decode_attention'];assert host['sha256']==c['source_sha256']['reports/baseline-integration-700-20261026/sgl_attention_recovery32/libraries/sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/attention/decode_attention.py']
  for f,v in c['oracle_source_sha256'].items():remote['/opt/tileops-all-pairs/workspace/'+f]=v
  remote[route['checker']]=c['verifier_sha256']
  rows.append({'key':key,'delta':delta,'checker':str(R/row['checker_receipt']),'guard':str(R/row['guard_receipt']),'guard_exit':0,'input_seed':c['input_seed'],'shape':expected,'verification':c['verification'],'native_events_checked':len(valid),'native_launch_count':n['native_launch_count'],'pid':pid,'binding_sha256':sha(R/delta/'dsa_adapter.py'),'qualified':True})
assert {tuple(r['key']) for r in rows}=={tuple(r[x] for x in ['baseline','operator','case_id']) for r in read(P/'SCOPE.json')}
(P/'AUDIT.json').write_text(json.dumps({'rows':rows,'evidence_hashes_verified':hashes,'diagnostic':'Only max_abs diagnostic tensor conversion moved to CPU; validator/reference unchanged.','boundary':'Full original decode_mla; selected causal set packing; page128, chunk64 queries; no kernel/schedule/compiler changes. Historical page16 prose is stale.','new_npu_runs':0},indent=2))
(P/'REMOTE_HASH_REQUEST.json').write_text(json.dumps(remote,indent=2))
(P/'HEARTBEAT.json').write_text(json.dumps({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'phase':'audit complete; registration pending','qualified':10}))
print('AUDIT PASS',len(rows),hashes)
