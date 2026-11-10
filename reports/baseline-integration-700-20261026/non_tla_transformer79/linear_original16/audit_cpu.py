"""CPU-only evidence projection: standard library, never imports providers or torch."""
import ast, collections, datetime, hashlib, json
from pathlib import Path
ROOT=Path('/root/dyq/workspace'); BASE=ROOT/'reports/baseline-integration-700-20261026'; OUT=BASE/'non_tla_transformer79/linear_original16'
def read(p): return json.loads(Path(p).read_text())
def ref(p,line=1):
 p=Path(p); p=p if p.is_absolute() else ROOT/p
 return {'path':str(p),'line':line,'exists':p.exists(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None}
scope=read(OUT/'SCOPE.json'); history=read(BASE/'surge_dense_linear92/RESULT.json'); hist={tuple(r['key']):r for r in history['rows']}
api='ascend_baselines/ops-transformer/attention/chunk_kda_fwd/op_host/op_api/aclnn_chunk_kda_fwd.h'
family={
 'linear':{'complete_public_api':'aclnnChunkKdaFwdGetWorkspaceSize / aclnnChunkKdaFwd','source':ref(api,25),'signature_source':ref(api,73),'capability':'FP16 Q/K/V; FP32 gate and beta; FP32 initial/final state; BSND, grouped value heads, INT64 sequence offsets, stateVFirst; both attention and final state. Scalar GDN log-gate broadcasts over K; ungated DeltaNet uses zero log-gate.','binding':ref(BASE/'surge_dense_linear92/worker05/package/binding.py',20),'alternatives':[{'name':'aclnnChunkGatedDeltaRule','status':'complete_api_unsupported','reason':'QKV, beta and output BF16 only; FP16 narrowing changes contract','source':ref('ascend_baselines/ops-transformer/attention/chunk_gated_delta_rule/op_host/op_api/aclnn_chunk_gated_delta_rule.cpp',55)},{'name':'RecurrentKda','status':'complete_api_unsupported','reason':'QKV BF16-only in actual tiling dtype validation','source':ref('ascend_baselines/ops-transformer/attention/recurrent_kda/op_host/recurrent_kda_tiling.cpp',243)},{'name':'aclnnRecurrentGatedDeltaRule','status':'complete_api_unsupported','reason':'query/key/value/beta and output BF16-only','source':ref('ascend_baselines/ops-transformer/attention/recurrent_gated_delta_rule/docs/aclnnRecurrentGatedDeltaRule.md',105)}]},
 'ssd':{'complete_public_api':'mambav2_chunk_state(dt_out,dacs,bt,xt) -> FP32 output','source':ref('ascend_baselines/ops-transformer/experimental/mamba/mamba2_chunk_state/torch_interface.cpp',96),'capability':'Native xt [B,C,L,H,P], bt [B,C,L,G,N], dt/dacs [B,C,L,H]; output [B,C,H,N,P]; original internally uses FP16 xt/bt and FP32 dt/dacs. Exact case needs transpose output N/P.','binding':ref(BASE/'transformer_linear/ssd_package/binding.py',16),'alternative':'No alternate complete FP16 chunk-state public API identified in current source review.'}}
rows=[]
for s in scope:
 key=tuple(s['key']);h=hist[key]; op=key[1];cap='32x8192' in key[2]
 c=h.get('checker_receipt') or h.get('checker') or h.get('prior_checker')
 if isinstance(c,dict):c=c.get('path')
 if not c or c=='failed':c=next((x['path'] for x in h.get('receipts',[]) if x['path'].endswith('/checker.json')),None)
 cp=Path(c) if c else None
 if cp and not cp.is_absolute():cp=ROOT/cp
 j=read(cp) if cp and cp.exists() else {}
 assert j, key
 params=j.get('case_snapshot',{}).get('parameters',{})
 numerical=j.get('status')=='failed' and 'Tensor-likes are not close' in str(j.get('error',''))
 if not cap:assert numerical,(key,j.get('status'),j.get('error'))
 refs=[ref(cp),ref(BASE/'surge_dense_linear92/RESULT.json'),ref(OUT/'DEPLOYMENT_CHECK.json')]
 snapshotline=next((i for i,l in enumerate(cp.read_text().splitlines(),1) if '"case_snapshot"' in l),1)
 refs[0]['line']=snapshotline
 native=h.get('native_receipt') or h.get('native');native=native if isinstance(native,str) else None
 if not native and (cp.parent/'dispatch.jsonl').exists():native=str(cp.parent/'dispatch.jsonl')
 contract_source='TileOPs/workloads/linear_attention/gdn.py' if op=='GDNFwdOp' else 'TileOPs/workloads/linear_attention/deltanet.py' if op=='DeltaNetFwdOp' else 'TileOPs/workloads/mamba.py'
 line=167 if op=='GDNFwdOp' else 357 if op=='DeltaNetFwdOp' else 769
 mapping={'parameters':params,'source':ref(contract_source,line),'seed':j.get('input_seed'),'outputs':['states'] if op=='SSDChunkStateFwdOp' else ['o','final_state']}
 if op=='GDNFwdOp':
  mapping['api_arguments']={'layout':'BSND','scale':'K**-0.5 when scale is null','gate':'g.float()[...,None].expand(B,S,HV,K).contiguous()','beta':'beta.float().contiguous()','stateVFirst':params.get('state_v_first',False),'cuSeqlens':'None for dense; unchanged cumulative offsets for ragged','initialState':'contiguous cloned FP32 input','outputs':'FP16 out, FP32 final; aqk/akk allocated as required auxiliary outputs'}
 elif op=='DeltaNetFwdOp':mapping['api_arguments']={'layout':'BSND','gate':'FP32 zeros [8,1,32,128]','beta':'lossless FP16 to FP32','initialState':'[8,32,128,128] FP32','outputs':'out [8,1,32,128] FP16 and final [8,32,128,128] FP32','native_argument_receipt':native}
 else:mapping['api_arguments']={'xt':'reshape [1,16,256,48,64]','bt':'reshape [1,16,256,1,128]','dt_dacs':'permute(0,2,3,1) -> [1,16,256,48]','output':'[1,16,48,128,64] transpose -> [1,16,48,64,128]','seq_idx':None}
 row={'key':list(key),'status':'deployment_blocked' if cap else 'original_numerical_failure','blocker_kind':'single_card_capacity' if cap else 'exact_original_checker_numerical_failure','family':'ssd' if op=='SSDChunkStateFwdOp' else 'linear','complete_api_exists':True,'contract':mapping,'checker':ref(cp),'checker_status':j.get('status'),'checker_error':j.get('error'),'historical_numerical_failure_verified':numerical,'native_receipt':native,'capture_directory':h.get('full_capture_remote',h.get('remote_captures')),'registration_pins':None,'verified_pending_publication':False,'deployment_evidence':str(OUT/'DEPLOYMENT_CHECK.json'),'changed_condition_proven':False,'retry_authorized':False,'next_legal_action':'Retain blocked; acceptance_root must first establish an equivalent complete original API mapping with a concrete mismatch versus this receipt. Do not rerun unchanged binding or change precision/tolerance.','sources':refs,'guard_budget_seconds':0,'new_device_execution':False}
 if cap:
  row['capacity_evidence']=ref(BASE/'surge_dense_linear92/worker04r2/RESOURCE_BOUND.json');row['next_legal_action']='No device submission on assigned 96-GiB host. Original full-check live lower bound 108.6587 GiB; even excluding gate/beta/aux outputs core bound 96.5962 GiB. Escalate physical capacity requirement; no checker clone removal, chunk splitting or workspace algorithm change.'
 rows.append(row)
# Check exact scope keys against registration modules without importing them.
hits=[]
for p in (ROOT/'TileOPs/benchmarks/ascend_all_pairs/ops_transformer').glob('integration*.py'):
 tree=ast.parse(p.read_text())
 for n in ast.walk(tree):
  if isinstance(n,ast.Constant) and isinstance(n.value,str) and n.value in {s['key'][2] for s in scope}:hits.append({'path':str(p),'line':n.lineno,'case_id':n.value})
result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'worker':'bd6d776cccb64a4c','scope_count':16,'accepted_added':0,'new_device_execution':False,'status_counts':dict(collections.Counter(r['status'] for r in rows)),'family_reviews':family,'rows':rows,'pending_publication_scan':{'source':'SCOPE.evidence -> exact original checker receipts, plus static current registration case literals','complete_success_count':0,'registration_literal_hits':hits},'provider_difference':{'current_default':ref('TileOPs/benchmarks/ascend_all_pairs/ops_transformer/adapters.py',472),'finding':'Default recurrent_family rejects FP16; historical private ChunkKdaFwd bypass already supplies the complete FP16 API. Registering it alone does not resolve numeric failure. SSD existing private binding already permutes input and transposes complete output correctly.','gdn_private_binding':ref(BASE/'surge_dense_linear92/worker05/package/binding.py',14)},'deployment_limit':'Current file hashes/exports verified on assigned remote only. Presence is not runtime success or proof current source built installed package; no device imports, checkers, benchmark or library loading performed.','adapter_candidate':None,'adapter_decision':'No evidenced semantic-equivalent adapter correction found; copying failed binding as cpu_ready would be misleading.','checks':['16 unique keys exactly equal scope','15 exact verify-stage numerical failures re-read from checker.json','capacity receipt re-read for remaining exact key','AST-only current provider registration scan','remote stdlib stat/hash + nm -D; no dlopen/import','source API/dtype/layout read against current code'],'remaining_issues':['No new legitimate retry condition established','Source-to-installed-package build identity remains inherited/unproven for official ChunkKdaFwd; preserve historical identity receipts','No executable handoff: guard budget zero; formal publication belongs to acceptance_root only']}
assert len(rows)==16 and len({tuple(r['key']) for r in rows})==16
(OUT/'RESULT.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
(OUT/'HEARTBEAT.json').write_text(json.dumps({'utc':result['utc'],'phase':'complete','status_counts':result['status_counts'],'npu_used':False},indent=2)+'\n')
print(json.dumps({'status_counts':result['status_counts'],'registration_literal_hits':hits,'rows':len(rows)}))
