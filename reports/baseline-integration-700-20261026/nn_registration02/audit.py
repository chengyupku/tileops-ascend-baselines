import pathlib,json,hashlib,re,datetime,sys
sys.dont_write_bytecode=True
W=pathlib.Path('/root/dyq/workspace'); D=W/'reports/baseline-integration-700-20261026'; P=D/'nn_registration02'; R=D/'nn_recovery02'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def save(n,v):(P/n).write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def evidence(p):return {'path':str(p),'sha256':sha(p)}
scope=json.loads((P/'SCOPE.json').read_text()); accepted={tuple(r['key']) for r in json.loads((D/'ACCEPTED.json').read_text())['rows']}
save('HEARTBEAT.json',{'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'phase':'auditing','npu_used':False})
compares=json.loads((R/'SOURCE_COMPARE.json').read_text())
for r in compares:
 assert r['identical'] and sha(r['source'])==sha(r['build'])==r['sha256'],r
compiles={}; all_events={}
for s in scope:
 cp=pathlib.Path(s['checker_receipt']);events=[json.loads(l) for l in (cp.parent/'native.jsonl').read_text().splitlines()];all_events[cp.parent.name]=events
 for e in events:
  if 'module[classicpilot_nn_impl.dynamic.mat_mul_v3]' in e['text'] and 'result[success]' in e['text']:
   k=re.search(r'kernel\[(te_matmulv3_[a-f0-9]+)\]',e['text'])[1];compiles[k]=e
rows=[];alerts=[]
for s in scope:
 key=[s['baseline'],s['operator'],s['case_id']]
 if tuple(key) in accepted:continue
 try:
  cp=pathlib.Path(s['checker_receipt']);gp=pathlib.Path(s['guard_receipt']);c=json.loads(cp.read_text());g=json.loads(gp.read_text());ap=cp.parent/'adapter.py';ah=sha(ap)
  assert c['status']=='verified' and g['returncode']==0
  assert g['requested_at']<=c['started_at']<c['finished_at']<=g['finished_at']
  assert c['verification']['checked_outputs']==c['verification']['total_outputs']==1
  assert c['inputs_preserved'] and not c['oracle_source_changed'] and not c['provider_source_changed']
  assert c['independent_cpu_reference'] and c['reference_device']=='cpu'
  assert c['source_sha256'][str(ap.relative_to(W))]==ah=='037f902f6779aa1d84dad1d5d04777e6754cec5e9b3d653b7339ac5f0dd200d7'
  checked={}
  # Hash only recorded original MatMul source and relevant runtime/oracle dependencies, never whole SDK.
  for k,v in {**c['source_sha256'],**c['oracle_source_sha256']}.items():
   if k.startswith('ascend_baselines/ops-nn/matmul/mat_mul_v3/') or k.startswith('TileOPs/') or ('classicpilot_nn/' in k and (k.endswith('.so') or 'mat_mul_v3' in k or k.endswith('aic-ascend950-ops-info.json'))):
    assert sha(W/k)==v,k;checked[k]=v
  dispatch=json.loads((cp.parent/'DISPATCH.json').read_text());assert dispatch['factory']==str(ap)+':make' and dispatch['registry_dispatch_asserted']
  gm=json.loads((cp.parent/'GUARD_METADATA.json').read_text());assert gm['owner']=='baseline700-20261026/nn_recovery02' and gm['phase']=='running'
  launches=[]
  for e in all_events[cp.parent.name]:
   t=e['text']
   if 'StreamLaunchKernelV1:' not in t or 'te_matmulv3_' not in t:continue
   m=re.search(r'(2026-10-26)-(\d\d:\d\d:\d\d)\.(\d{3})\.(\d{3})',t);stamp=m[1]+'T'+m[2]+'.'+m[3]+m[4]+'+00:00';assert c['started_at']<=stamp<=c['finished_at']
   kernel=re.search(r'te_matmulv3_[a-f0-9]+',t)[0];assert kernel in compiles
   ce=compiles[kernel];cm=re.search(r'(2026-10-26)-(\d\d:\d\d:\d\d)\.(\d{3})\.(\d{3})',ce['text']);compile_stamp=cm[1]+'T'+cm[2]+'.'+cm[3]+cm[4]+'+00:00';assert compile_stamp<=stamp
   launches.append({'kernel':kernel,'launch':e,'compile':ce,'at':stamp})
  assert launches
  # Independently verify extracted events against original line-numbered raw logs.
  for e in [x for pair in launches for x in [pair['launch'],pair['compile']]]:
   raw=pathlib.Path(e['file']);line=next(l.rstrip('\n') for i,l in enumerate(raw.open(errors='replace'),1) if i==e['line']);assert line==e['text']
  rows.append({'key':key,'status':'reviewed','checker':evidence(cp),'guard':evidence(gp),'guard_metadata':evidence(cp.parent/'GUARD_METADATA.json'),'actual_tested_factory':dispatch['factory'],'adapter_sha256':ah,'full_outputs':c['verification'],'input_seed':c['input_seed'],'case_snapshot':c['case_snapshot'],'oracle_source_sha256':c['oracle_source_sha256'],'source_runtime_hashes_checked':checked,'source_compare':evidence(R/'SOURCE_COMPARE.json'),'native':evidence(cp.parent/'native.jsonl'),'owned_events':launches,'runtime_host':c['runtime_host'],'semantic_review':'Original complete MatMulV3 graph API; trans_a/trans_b views, contiguous FP32 dtype binding and output cast only. No new arithmetic/kernel/schedule. Complete single output, canonical seed and independent CPU oracle. classicpilot_nn dynamic compile and matching native kernel launch; final fp16 case reuses preceding fp32 graph kernel cache.'})
 except Exception as e:alerts.append({'key':key,'status':'REJECTED','reason':repr(e),'trace':__import__('traceback').format_exc()})
save('AUDIT.json',{'rows':rows,'source_comparison_files':len(compares),'missing_optional_packets':[n for n in ['merge_packet','MERGE_MAP.json','ATTRIBUTION_SUMMARY.json'] if not (R/n).exists()]});save('ALERTS.json',alerts)
print('reviewed',len(rows),'rejected',alerts)
