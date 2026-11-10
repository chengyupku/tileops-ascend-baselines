"""CPU-only source accounting. No framework import, native load, case materialization or execution."""
from pathlib import Path
import ast, datetime, hashlib, json, math, re, runpy, shutil, sys
W=Path('/root/dyq/workspace'); D=Path(__file__).resolve().parent; T=D.parent/'bf16_moe_hadamard_native10'; G=1024**3
stamp=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(name,data): (D/name).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
def key(r):return (r['baseline'],r['operator'],r['case_id'])
scope=json.loads((D/'SCOPE.json').read_text()); full=json.loads((T/'SCOPE.json').read_text())
assert len(scope)==len({key(r) for r in scope})==6
assert len(full)==len({key(r) for r in full})==10
assert set(map(key,scope))<=set(map(key,full))
old={tuple(r['key']):r for r in json.loads((T/'CASES.json').read_text())}
results=json.loads((T/'RESULT.json').read_text()); verified={tuple(r['key']) for r in results['rows'] if r['status']=='verified'}
assert set(map(key,scope))==set(map(key,full))-verified
manifest=W/'TileOPs/src/tileops/manifest/spec/moe.yaml'
lines=manifest.read_text().splitlines(); section=lines[320:330]
# Scope drives selection; dimensions come from corresponding original manifest row, never a copied case list.
rows=[]
for s in scope:
 k=key(s); cid=k[2]; label=cid.removesuffix('-bfloat16-float32').removesuffix('-bfloat16'); fp32=cid.endswith('-float32')
 matches=[(i+321,l) for i,l in enumerate(section) if f'"{label}"' in l and ('out_dtype: float32' in l)==fp32]
 assert len(matches)==1,(cid,matches)
 line,l=matches[0]; dims={a:int(b) for a,b in re.findall(r'\b(P|K|E|N|max_m): (\d+)',l)}
 E,K,N=dims['E'],dims['K'],dims['N']; masked='max_m' in dims; R=E*dims['max_m'] if masked else dims['P']
 counts=[dims['max_m'] if e%2==0 else dims['max_m']//2 for e in range(E)] if masked else [R//E+(e<R%E) for e in range(E)]
 metadata=counts if masked else [sum(counts[:i+1]) for i in range(E)]
 V=sum(counts); M=max(counts); J=sum(c>0 for c in counts); A=R*K; B=E*N*K; Y=R*N; outbytes=4 if fp32 else 2
 I=2*(A+B)+4*E; O=outbytes*Y; ret=4*(V*K+J*N*K+V*N)
 params={'a':[[E,dims['max_m'],K] if masked else [R,K],'bfloat16'],'b':[[E,N,K],'bfloat16'],'layout_metadata':[[E],'int32'],'output':[[E,dims['max_m'],N] if masked else [R,N],'float32' if fp32 else 'bfloat16'],'layout':{'masked':{'max_m':dims['max_m']}} if masked else {'contiguous':{'packing':'tight','metadata_kind':'physical_psum','alignment':1}},'activation':None,'out_dtype':'float32' if fp32 else None}
 S=A+B+Y; envelope={'host_required':2*G+64*S,'device_required':2*G+32*S,'disk_required':G+8*S}
 if k in old:
  assert old[k]['manifest_params']==params,(cid,'params mismatch')
  assert old[k]['envelope']==envelope
 # Upper bounds for explicit tensor payloads, NOT opaque framework/native allocation bounds.
 base=2*I+3*O+ret+R
 phases={
 'casegen':{'host_explicit_payload_upper_bytes':4*E,'hbm_explicit_payload_upper_bytes':max(6*A,2*A+6*B+4*E),'formula':'max(6*A,2*A+6*B+4*E); randn default FP32 then BF16, outputs skipped','source':['TileOPs/src/tileops/manifest/workload.py:133-160','TileOPs/workloads/moe.py:265-268']},
 'snapshot':{'host_explicit_payload_upper_bytes':0,'hbm_explicit_payload_upper_bytes':2*I+R,'formula':'2*I + R mask; original full inputs plus pristine clone','source':['TileOPs/workloads/numerics.py:525-526','TileOPs/workloads/numerics.py:427-428']},
 'cpu_reference':{'host_explicit_payload_upper_bytes':I+4*Y+4*(M*K+N*K+M*N)+O+R,'hbm_explicit_payload_upper_bytes':2*I+2*O+R,'formula':'host I+4*Y+4*(M*K+N*K+M*N)+O+R; device 2*I+2*O+R. Conservative sum includes cast output and one expert FP32 operands/product even if not simultaneous','source':['TileOPs/scripts/ascend_pair_check.py:137-159','TileOPs/workloads/moe.py:239-259','TileOPs/workloads/numerics.py:530-531']},
 'candidate_and_output_clone':{'host_explicit_payload_upper_bytes':120*J+4*E,'hbm_explicit_payload_upper_bytes':2*I+4*O+ret+2*N*K+outbytes*M*N+R,'formula':'2*I+4*O+ret+2*N*K+d*M*N+R; ret=4*(V*K+J*N*K+V*N). Entire original/pristine weights and every native FP32 operand/output coexist; BF16 contiguous weight and output cast transient included','source':['../bf16_moe_hadamard_parent10/adapter.py:31-71','../bf16_local_gemm18/binding.py:13-30','TileOPs/benchmarks/ascend_math_more.py:58-102','TileOPs/workloads/numerics.py:552']},
 'input_preservation':{'host_explicit_payload_upper_bytes':120*J+4*E,'hbm_explicit_payload_upper_bytes':base,'formula':'2*I+3*O+ret+R PLUS opaque assert_close temporaries for each full input, largest B elements; no blockwise shortcut','source':['TileOPs/workloads/numerics.py:443-452','TileOPs/workloads/numerics.py:553-554']},
 'full_defined_rows_checker':{'host_explicit_payload_upper_bytes':120*J+4*E,'hbm_explicit_payload_upper_bytes':base+(2*outbytes+8)*V*N+16*R,'formula':'base+(2*d+8)*V*N+16*R index allowance; both selected copies and FP32 casts conservatively overlap. PLUS assert_close backend temporaries','source':['TileOPs/workloads/moe.py:279-296','TileOPs/workloads/numerics.py:353-357']},
 'metrics':{'host_explicit_payload_upper_bytes':120*J+4*E,'hbm_explicit_payload_upper_bytes':base+64*min(Y,1<<20),'formula':'base+64*min(Y,2**20); conservative payload allowance for block FP64 operands, boolean masks, selections, subtraction/abs. Opaque backend scratch excluded','source':['TileOPs/workloads/numerics.py:366-383']}}
 h=max(p['host_explicit_payload_upper_bytes'] for p in phases.values()); b=max(p['hbm_explicit_payload_upper_bytes'] for p in phases.values())
 rows.append({'key':list(k),'history_status':'admissionblocked_before_input_allocation' if cid=='ds-v3-decode-down-bfloat16' else 'never_run_in_native10','historical_numerical_failure':False,'missing_from_native10_cases':k not in old,'manifest_source':f'TileOPs/src/tileops/manifest/spec/moe.yaml:{line}','manifest_params':params,'metadata_values':metadata,'metadata_source':'TileOPs/src/tileops/manifest/primitives.py:499-510','E':E,'K':K,'N':N,'materialized_rows':R,'defined_rows':V,'max_rows_per_expert':M,'native_calls_first_fullcheck':J,'original_envelope':envelope,'envelope_provenance':'original CASES.json exact match' if k in old else 'not historically configured; same original prepare.py formula reconstructed for omitted SCOPE key','elements':{'A':A,'B':B,'Y':Y},'bytes':{'I':I,'O':O,'retained_native_tensor_payload':ret,'explicit_ctypes_shape_stride_pointer_arrays':120*J},'phases':phases,'peak_explicit_payload_upper':{'host_bytes':h,'hbm_bytes':b},'planning_reserves_not_gate_changes':{'host_fixed_bytes':2*G,'hbm_fixed_bytes':2*G,'host_allocator_25pct_explicit_bytes':math.ceil(h/4),'hbm_allocator_25pct_explicit_bytes':math.ceil(b/4),'disk_floor_bytes':2*G},'total_peak_certified':False,'total_peak_expression':{'host':'max(host explicit phases)+CPU BLAS scratch+Python/ACL descriptor/executor+host allocator+2GiB','hbm':'max(HBM explicit phases)+sum(workspace_i for all J calls)+framework op temporaries+allocator retained/reserved blocks+2GiB'},'unknowns':['GetWorkspaceSize return per expert, sum retained to cleanup','full-input torch.testing.assert_close/isclose temporaries and backend workspaces','Custom full defined-row assert_close backend temporaries','CPU BLAS scratch, runtime/descriptor/executor overhead','allocator fragmentation/cached blocks; 25% is a budget, not a source-proven maximum'],'ready_without_lowering_existing_gate':masked,'new_execution_condition':'Fresh guarded original per-key host+cgroup/HBM/disk admission; independent masked first-fullcheck; native provenance and entire checker receipt required' if masked else 'Original per-key gate remains binding; insufficient historical HBM for existing envelope. Any tighter gate needs separately justified unknown budgets and parent decision; omitted key additionally needs SCOPE-complete preparation.'})
capacity=runpy.run_path(str(T/'resource_gate.py'))['capacity']()
capacity.update(captured_at=stamp(),disk_free_bytes=shutil.disk_usage(D).free,hbm_read_this_task=False)
historical=json.loads((T/'ds-v3-decode-down-bfloat16/RESOURCE_ADMISSION.json').read_text())
for r in rows:
 env=r['original_envelope'];r['capacity_comparison']={'fresh_host_cgroup_sufficient_original_gate':capacity['effective_available_host_bytes']>=env['host_required'],'fresh_disk_sufficient_original_gate':capacity['disk_free_bytes']>=env['disk_required'],'historical_hbm_free_sufficient_original_gate':historical['hbm_free']>=env['device_required'],'fresh_hbm_status':'NOT_READ_CPU_ONLY','historical_hbm_is_current_admission':False}
dump('CAPACITY_CPU.json',capacity)
dump('PEAK_MEMORY_PLAN.json',{'created_at':stamp(),'accepted_added':0,'cpu_only':True,'runs':0,'scope_count':6,'full_scope_count':10,'scope_driven':True,'original_prepared_count':len(old),'scope_coverage_assertions':'SCOPE6 = SCOPE10 minus four existing verified receipts; exact unique keys; all five available CASES rows params/envelopes match original source','historical_capacity_receipt':str(T/'ds-v3-decode-down-bfloat16/RESOURCE_ADMISSION.json'),'historical_hbm_free_bytes':historical['hbm_free'],'historical_hbm_total_bytes':historical['hbm_total'],'status':'conditional_explicit_tensor_payload_bounds_not_replacement_admission','scope_rows':rows})
dump('HEARTBEAT.json',{'at':stamp(),'status':'CPU source accounting complete; validating delivery','accepted_added':0,'npu_calls':0})
assert not any(n=='torch' or n.startswith('torch_npu') for n in sys.modules)
print(json.dumps([{'case':r['key'][2],'host_payload_GiB':round(r['peak_explicit_payload_upper']['host_bytes']/G,3),'hbm_payload_GiB':round(r['peak_explicit_payload_upper']['hbm_bytes']/G,3),'old_host_GiB':r['original_envelope']['host_required']/G,'old_hbm_GiB':r['original_envelope']['device_required']/G,'gate_candidate':r['ready_without_lowering_existing_gate']} for r in rows],indent=2))
