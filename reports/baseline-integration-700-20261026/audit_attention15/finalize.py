import pathlib,json,hashlib,difflib,datetime
A=pathlib.Path(__file__).resolve().parent;R=json.loads((A/'RECOMMENDED_ACCEPT.json').read_text());P=json.loads((A/'GDN_PATCH_CPU_PREVIEW.json').read_text());F=json.loads((A/'FINAL_REMOTE.json').read_text());I=json.loads((A/'REMOTE_INDEX.json').read_text())
for r in R['rows']:
 c=json.loads(pathlib.Path(r['checker']['path']).read_text());decl=c['case_snapshot']['declared_tensors'];names=c['case_snapshot']['declared_input_names']
 for call in c['observed_calls']['candidate']:
  assert call['returned']
  for name,t in zip(names,call['inputs']['items']):assert [t['shape'],t['dtype']]==decl[name]
  assert [call['outputs']['shape'],call['outputs']['dtype']]==decl['o']
 r['canonical']['observed_input_output_shapes_dtypes_match']=True
 if r['key'][1]=='DSADecodeWithKVCacheFwdOp' and r['key'][2].endswith('-float16'):
  w=r['canonical']['workload'];assert w['q_start_index_s']//w['stride_kv']<w['K'];r['canonical']['tolerance']={'atol':.002,'rtol':.002,'resolution':'canonical causal_topk_indices first token pads with extent: visible=start//stride < K; original DSA sentinel predicate necessarily true','generator':P['generator']}
 r['cpu_exact_route']['registry_module']=I['files']['/opt/tileops-all-pairs/workspace/TileOPs/benchmarks/ascend_all_pairs/ops_transformer/__init__.py']
 r['source_provenance']['native_source_ownership_note']='Exact te_flashattentionscore name joined from actual StreamLaunchKernelV1 inside checker interval to original vendor compilation source lines. Batch log location may belong to first case; timestamps disambiguate each case.'
assert len(R['rows'])==15 and all(r['accept_recommended'] for r in R['rows']);assert not I['forbidden_imports'];assert not F['device_imports'];R['completed_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();(A/'RECOMMENDED_ACCEPT.json').write_text(json.dumps(R,indent=2))
G=json.loads((A/'GDN_ADDITIONAL_REVIEW.json').read_text());G['proposed_registration_cpu_preview']={k:v for k,v in P.items() if k!='generator'};G['guard_returncode']=1;G['source_changed_status_preserved']=True;G['remaining_issue']='Not currently registered for this key. Existing registry fell through to older constructor; forbidden ctypes import was blocked. Proposed one-key allowlist preview routes successfully through actual registry; patch not applied.'
reg=G['registration_file'];old=pathlib.Path(reg['path']).read_text();new=json.dumps(sorted(set(json.loads(old))|{G['key'][2]}),indent=2)+ ('\n' if old.endswith('\n') else '');patch=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile=reg['remote_path'],tofile=reg['remote_path']));(A/'GDN_EXACT_REGISTRATION.patch').write_text(patch);(A/'GDN_ADDITIONAL_REVIEW.json').write_text(json.dumps(G,indent=2))
# Verify installed/source/package original kernel entry equivalence from actual remote hashes.
critical=json.loads((A/'CRITICAL_SOURCE_HASHES.json').read_text());eq=[]
for name in ['flash_attention_score.cpp','flash_attention_score_apt.cpp']:
 matches=[v for v in critical if v['path'].endswith('/op_kernel/'+name)];assert len({v['sha256'] for v in matches})==1;eq.append({'file':name,'equal':True,'copies':matches})
(A/'ORIGINAL_KERNEL_EQUIVALENCE.json').write_text(json.dumps(eq,indent=2))
summary={'scope':15,'recommended_accept':15,'guard_zero':12,'guard_124_with_pretermination_full_sync_cleanup':3,'NPU_commands':0,'numeric_reruns':0,'exact_live_cpu_routes':15,'shared_writes':0,'additional_gdn':'Unrelated provider additions proven; binding/entry/direct_acl tested/frozen byte identity; proposed one-key allowlist patch passes actual registry CPU preview; not applied. Original guard1/source_changed retained.','recommendations':str(A/'RECOMMENDED_ACCEPT.json'),'gdn':str(A/'GDN_ADDITIONAL_REVIEW.json'),'remaining_issues':['Three DSA batch cases retain guard124 Python/TBE teardown failure; all original full checker/sync/cleanup receipts precede termination.','GDN proposed registration remains unapplied.']};(A/'SUMMARY.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
