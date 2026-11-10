from pathlib import Path
import json,hashlib,datetime
P=Path(__file__).resolve().parent; W=Path('/root/dyq/workspace');B=P.parents[1];scope=json.loads((P/'SCOPE.json').read_text());rows=[]
for idx in [0,6,7,8,9,10,11,12]:
 x=scope[idx];r={'key':x['key'],'status':x['status'],'reason':x['reason'],'registered':False,'accepted_added':0,'executed_this_task':False,'attempts':[],'prior_scope_path':x['prior_scope_path'],'retry_condition_changed':False,'runtime_ready':False}
 if idx==0: paths=[B/'sgl_gdn_remote38'/x['checker_receipt'],B/'sgl_gdn_remote38'/x['guard_receipt']];r['status']='oracle_blocked_before_candidate'
 elif idx==11: paths=[Path(x['checker_receipt']),Path(x['guard_receipt']),Path(x['parent_precision_review'])];r['status']='oracle_blocked_before_candidate'
 elif idx==12: paths=[Path(x['prior_checker']),Path(x['prior_checker']).parent/'GUARD_RESULT.json'];r['status']='original_numerical_failure'
 elif idx==6:
  paths=[W/'TileOPs/src/tileops/manifest/spec/linear_attention.yaml',W/'ascend_baselines/sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/fla/chunk.py',W/'ascend_baselines/sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/fla/fused_sigmoid_gating_recurrent.py',W/'ascend_baselines/sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/fla/fused_sigmoid_gating_recurrent_decode_optimized.py']
  r['reason']='Exact case raw gate=true, beta sigmoid=false (manifest default false, line 128 case); native chunk API has no raw gate interface. Original fused recurrent APIs implement raw gate but unconditionally sigmoid beta (fused_sigmoid_gating_recurrent.py:120 and decode_optimized.py:96). Using either changes exact-case math; no kernel edits or inverse-sigmoid replacement allowed.'
 else: paths=[Path(x['latest_worker_report']),W/'TileOPs/benchmarks/ascend_all_pairs/ops_transformer/selection_values.py',W/'ascend_baselines/ops-transformer/moe/moe_gating_top_k/op_host/moe_gating_top_k_tiling.cpp']
 r['reused_evidence']=[{'path':str(p),'exists':p.exists(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None} for p in paths]
 d=P/('retained'+str(idx));d.mkdir(exist_ok=True);(d/'RESULT.json').write_text(json.dumps(r,indent=2));rows.append(r)
(P/'RETAINED_RESULTS.json').write_text(json.dumps(rows,indent=2))
(P/'HOST_RECIPE.json').write_text(json.dumps({'task':'actual exact-case supplemental checking','selected_host':'openlibing-DevEnv-055409','architecture':'aarch64','python':'/opt/tileops-all-pairs/bin/allpairs-python','python_abi':'3.12 (reused host runtime recipe)','cann':'/usr/local/Ascend/cann-9.1.0','original_source':'/opt/tileops-all-pairs/build/sgl-pristine-20261025T2310/source','prior_cpu_preflight':str(B/'sgl_gdn_remote38/CPU_PREFLIGHT.json'),'delta':'private copied wrapper paths only; no kernel/compiler/checker/reference changes','server1':'aarch64 runtime available; exact pristine SGL source path absent','local':'busy at initial preflight; selected existing compatible server2 deployment','clock_note':'remote wall clock approximately 94 seconds behind local; all per-attempt durations derived on same remote host','runtime_ready':False},indent=2))
print('Retained',len(rows),'exact results; no device replay')
