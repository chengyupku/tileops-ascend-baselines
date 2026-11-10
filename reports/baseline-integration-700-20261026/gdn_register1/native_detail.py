import pathlib,json,hashlib,subprocess
W=pathlib.Path('/opt/tileops-all-pairs/workspace');D=W/'reports/baseline-integration-700-20261026/gdn_register1';A=D/'attempt_003_native';A.mkdir(exist_ok=False);T=D.parent/'transformer_linear';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
n=json.loads((D/'attempt_001/NATIVE.json').read_text());vendor=pathlib.Path(n['recipe']['vendor']);original=W/'.cache/all-pairs/root_server2_20261022/source_templates/ascend_baselines/ops-transformer/attention/chunk_gated_delta_rule';matches=[]
for p in (vendor/'op_impl/ai_core/tbe/kdapairs_transformer_impl/ascendc/chunk_gated_delta_rule/op_kernel').rglob('*'):
 if p.is_file():
  rel=p.relative_to(vendor/'op_impl/ai_core/tbe/kdapairs_transformer_impl/ascendc/chunk_gated_delta_rule');q=original/rel
  matches.append({'installed':str(p),'original':str(q),'sha256':sha(p),'original_sha256':sha(q) if q.exists() else None,'equal':q.exists() and sha(p)==sha(q)})
assert all(x['equal'] for x in matches)
out={'kernel_source_equivalence':matches,'batch_logs':[],'build_logs':[]}
for name in ['guard.stdout.log','guard.stderr.log']:
 p=T/'batch_2'/name;txt=p.read_text(errors='replace');(A/name).write_text(txt);out['batch_logs'].append({'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size})
# This is the precise historical build owning the loaded opapi and pinned device objects.
build=vendor.parents[3]
out['build_root']=str(build)
for p in list(build.glob('*.log'))+list(build.parent.glob('*.log')):
 lines=p.read_text(errors='replace').splitlines();hits=[{'line':i+1,'text':s} for i,s in enumerate(lines) if any(k in s.lower() for k in ['chunk_gated_delta_rule','chunkgateddeltarule'])]
 if hits:out['build_logs'].append({'path':str(p),'sha256':sha(p),'hits':hits[-35:]})
# Official host API and original example, exact ABI compared with binding transport.
for name,p in [('official_api',original/'op_host/op_api/aclnn_chunk_gated_delta_rule.cpp'),('official_header',original/'op_host/op_api/aclnn_chunk_gated_delta_rule.h'),('official_example',original/'examples/test_aclnn_chunk_gated_delta_rule.cpp')]:
 if p.exists():
  (A/(name+p.suffix)).write_bytes(p.read_bytes());lines=p.read_text().splitlines();out[name]={'path':str(p),'sha256':sha(p),'abi_lines':[{'line':i+1,'text':s} for i,s in enumerate(lines) if any(k in s for k in ['aclnnChunkGatedDeltaRule','workspaceSize','output','finalState','stateOut'])]}
(A/'NATIVE_DETAIL.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'kernel_source_files_equal':len(matches),'build_logs':len(out['build_logs']),'batch_logs':out['batch_logs']}))
