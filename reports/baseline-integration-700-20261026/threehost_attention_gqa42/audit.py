from pathlib import Path
import ast,json,yaml,hashlib,datetime,collections,subprocess
W=Path('/root/dyq/workspace'); B=W/'reports/baseline-integration-700-20261026'; P=B/'threehost_attention_gqa42'; V=W/'ascend_baselines/tilelang-ascend'; O=B/'tla_attention_recovery01'; E=O/'original_remote_evidence'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(n,v):(P/n).write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
scope=json.loads((P/'SCOPE.json').read_text()); manifest=W/'TileOPs/src/tileops/manifest/spec/attention.yaml'; m=yaml.safe_load(manifest.read_text())
specs=[('expert','examples/gqa_fwd_varlen/gqa_fwd_varlen.py','flashattn','qwen3-30b-1k-float16-pto-v3'),('direct','examples/flash_attention/flash_attn_bhsd.py','flash_attention_fwd','gemma-2b-8k-bs1-float16-pto-v3'),('sink','examples/attention_sink/example_gqa_sink_fwd_varlen/example_gqa_sink_fwd_varlen.py','gqa_sink_fwd','serving-kv512-float16-pto-v2')]
routes={}
for route,src,func,run in specs:
 tree=ast.parse((V/src).read_text()); f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==func)
 old=json.loads((E/'runs'/run/'native.json').read_text()); oldhash=next(v for k,v in old['hashes'].items() if k.endswith(src)); assert oldhash==sha(V/src)
 defaults={a.arg:ast.unparse(d) for a,d in zip(f.args.args[-len(f.args.defaults):],f.args.defaults)} if f.args.defaults else {}
 const={t.id:ast.unparse(n.value) for n in tree.body if isinstance(n,ast.Assign) for t in n.targets if isinstance(t,ast.Name) and t.id in ['PASS_CONFIGS','pass_configs','NUM_CORES']}
 kernel=next((E/'runs'/run/'jit/launch-evidence').glob('*/*.kernel.txt'))
 routes[route]=dict(source=src,function=func,source_sha256=oldhash,line=f.lineno,decorators=[ast.unparse(d) for d in f.decorator_list],defaults=defaults,explicit_constants=const,target='pto',platform='A5',original_decorator_target='auto (public PTO selection admitted)',prior_exact_run=run,prior_native=str(E/'runs'/run/'native.json'),generated_kernel=str(kernel),generated_sha256=sha(kernel),source_unchanged=True)
 routes[route]['pass_configs']=({k:False for k in ['tl.ascend_auto_cv_combine','tl.ascend_auto_cv_sync','tl.ascend_auto_sync','tl.ascend_memory_planning']} if route!='direct' else {})
 routes[route]['indices']=dict(out_idx=[4] if route=='expert' else [3] if route=='direct' else [8],workspace_idx=[5,6,7] if route=='expert' else [4,5,6] if route=='direct' else [9,10,11])
 routes[route]['compile_call_evidence']=str(O/'original_metadata/adapter.py')+':5-9 (PTO plus original jit metadata)'
 routes[route]['historical_platform_evidence']='Generated __DAV_C310_CUBE__/__DAV_C310_VEC__ and PTO_PLATFORM_A5; original_metadata/run_case.py uses TL_PLATFORM=A5'
 routes[route]['fresh_loaded_claim']=False
 routes[route]['closure_limit']='Current source hashes equal old run receipts; original Python closure comparison has 144 equal files in prior pto73 CPU audit. Full immutable historical SDK/header closure unavailable; no new dependency remedy established.'

def lens(v):
 if isinstance(v,list):return v
 c=ast.parse(v,mode='eval').body;assert isinstance(c,ast.Call) and c.func.id=='repeat'; a,b=[ast.literal_eval(x) for x in c.args];return [a]*b
rows=[]
for x in scope:
 op=x['operator'];cid=x['case_id']; w=max((w for w in m[op]['workloads'] if cid.startswith(w['label']+'-')),key=lambda w:len(w['label']))
 d=w['D']; route='expert' if d==128 else 'sink' if d<128 else 'direct'; cfg=routes[route]
 ql=[w['S_q']] if op=='GQADenseFwdOp' else lens(w['q_lens']);kl=[w['S_kv']] if op=='GQADenseFwdOp' else lens(w['kv_lens'] if op=='GQAVarlenFwdOp' else w['cache_lens'])
 unsupported=d>128 and any(q!=1 for q in ql)
 bindings=[]
 for q,k in sorted(set(zip(ql,kl))):
  if not q:bindings.append({'q_len':0,'action':'shared paged adapter skips empty query; old private adapter does not, so must retain shared behavior in future binding'});continue
  b=w['B'] if op=='GQADenseFwdOp' else 1
  args=[b,w['H']//w['H_kv'],w['H'],d,((q+127)//128)*128,((k+127)//128)*128,w.get('is_causal',True)] if route=='expert' else [b,k,w['H'],d] if route=='direct' else [b,w['H']//w['H_kv'],w['H'],128,((q+63)//64)*64,((k+127)//128)*128,w.get('is_causal',True)]
  bindings.append({'logical_q':q,'logical_kv':k,'positional_builder_args':args,'kwargs':{'sm_scale':d**-.5,'mask_tiles':b*((q+63)//64)*((k+127)//128)} if route=='sink' else {},'valid_binding':not unsupported})
 prior=cid in ['qwen3-30b-1k-float16','gemma-2b-8k-bs1-float16']
 status='prior_exact_device_timeout_preserved' if prior else 'input_contract_blocked' if unsupported else 'unverified_same_source_route_expansion_gated'
 rows.append(dict(key=x['key'],operator=op,case_id=cid,manifest_workload=w,manifest_sha256=sha(manifest),source=cfg['source'],function=cfg['function'],source_sha256=cfg['source_sha256'],route=route,pass_configs=cfg['pass_configs'],indices=cfg['indices'],builder_defaults=cfg['defaults'],target='pto',platform='A5',bindings=bindings,status=status,executed_this_task=False,accepted_added=0,numerical_success=False,numerical_failure_established=False,prior_exact_timeout=cfg['prior_exact_run'] if prior else None,route_representative_timeout=cfg['prior_exact_run'],reason='Direct source has no causal/window/mask argument; current binding explicitly requires sq=1. Multirow causal varlen cannot be credited or run unchanged through this binding.' if unsupported else 'Representative of this exact source and explicit configuration has actual device noncompletion; no changed dependency/configuration remedy. This key is not inferred numerically failed.' if not prior else 'Historical exactkey source/config already timed out; no unchanged rerun.',host_adaptation=('BSHD/BHSD pad/crop' if op=='GQADenseFwdOp' else 'current offsets slice independent sequences' if op=='GQAVarlenFwdOp' else 'current page table/cache lengths gather; independent dense sequence'),window=[w.get('window_size_left',-1),w.get('window_size_right',-1)],shared_adapter_evidence='TileOPs/benchmarks/ascend_all_pairs/tla/adapters.py:400-469,756-801; plans.py:191-207',private_extension_evidence=str(O/'original_metadata/adapter.py')+':25-88'))
assert len(rows)==42 and len({tuple(r['key']) for r in rows})==42
dump('EXACTKEY_SOURCE_MAPPING.json',rows);dump('ROUTES.json',routes)
dump('RESULT.json',rows)
checks={str(p):sha(p) for p in [manifest,W/'TileOPs/scripts/ascend_pair_check.py',W/'TileOPs/workloads/numerics.py',W/'TileOPs/benchmarks/ascend_all_pairs/tla/adapters.py',W/'TileOPs/benchmarks/ascend_all_pairs/tla/native.py',W/'TileOPs/benchmarks/ascend_all_pairs/tla/plans.py',O/'original_metadata/adapter.py',O/'original_metadata/run_case.py']}
dump('READONLY_INPUT_HASHES.json',checks)
summary=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=42,accepted_added=0,device_runs=0,route_counts=dict(collections.Counter(r['route'] for r in rows)),statuses=dict(collections.Counter(r['status'] for r in rows)),new_untested_admissible_source_routes=0,conclusion='Independently audited expert GQA, plain flash attention, and sink sources. Two in-scope exact timeouts preserved; 37 unrun keys remain unverified behind their own source representative; three multiquery D512 varlen keys lack a valid current original API binding. No sink-only extrapolation.',checks=['42 unique scope keys matched literal manifest workloads','AST read of each original builder/default/decorator/pass constant','Three source hashes matched their respective historical native receipts and both remote CPU probes','Prior compiled source/ELF/stackcore inspected; no fresh runtime/compiler load claimed','No original kernel/compiler/checker/registry/memory writes'],remaining=['No complete checker success for scope; parent registration must remain zero','No device-PC/event snapshot in old host stackcore; synchronization defect remains unproven','A new independently established original deployment fix or diagnostic authorization is needed before replaying timed-out routes'],files=['RESULT.json','EXACTKEY_SOURCE_MAPPING.json','ROUTES.json','SYNC_DIAGNOSIS.json','READONLY_INPUT_HASHES.json','DELIVERY.json','FILEHASHES.json','HEARTBEAT.json'])
dump('DELIVERY.json',summary)
dump('HEARTBEAT.json',dict(at=summary['at'],phase='completed_source_audit_no_new_admissible_route',accepted_added=0,device_runs=0))
print(json.dumps(summary,indent=2))
