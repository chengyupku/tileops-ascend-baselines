from pathlib import Path
import json,hashlib,datetime,socket,difflib,ast
R=Path(__file__).resolve().parent; W=R.parents[2]
result=json.loads((R/'RESULT.json').read_text()); rows=result['rows']; scope=json.loads((R/'SCOPE.json').read_text())
assert len(rows)==9 and all(x['status']=='verified_unregistered' for x in rows)
module=R/'integration700_sgl_conv_sampling02.py';digest=hashlib.sha256(module.read_bytes()).hexdigest()
for row in rows:
 c=json.loads(Path(row['checker_receipt']).read_text()); assert c['source_sha256'][str(module.relative_to(W))]==digest
 assert c['verification']['checked_outputs']==c['verification']['total_outputs']
 assert not c.get('provider_source_changed') and not c.get('oracle_source_changed')
 row['source_sha256']=c['source_sha256'];row['max_abs_err']=c['verification']['max_abs_err']
 for path,h in c['source_sha256'].items():
  assert hashlib.sha256((W/path).read_bytes()).hexdigest()==h,path
conv=[x['case_id'] for x in rows if x['operator']=='Conv2dFwdOp'];sampling=[x['case_id'] for x in rows if x['operator']=='ChainSpeculativeSamplingFwdOp']
fragment='\n# Exact keys privately verified by baseline700 sgl_conv_sampling02.\nfrom .integration700_sgl_conv_sampling02 import conv as _i700_cs02_conv, chain as _i700_cs02_chain\n\ndef _i700_cs02_dispatch(previous, factory, ids):\n    def build(case):\n        return factory(case) if case.id in ids else previous(case)\n    return build\n'
for op,fn,ids in [('Conv2dFwdOp','_i700_cs02_conv',conv),('ChainSpeculativeSamplingFwdOp','_i700_cs02_chain',sampling)]:
 fragment+=f"PROVIDERS[{op!r}] = _i700_cs02_dispatch(PROVIDERS[{op!r}], {fn}, {tuple(ids)!r})\n"
(R/'REGISTRATION_APPEND.py').write_text(fragment)
provider=W/'TileOPs/benchmarks/ascend_all_pairs/sgl/__init__.py'; original=provider.read_text();target='TileOPs/benchmarks/ascend_all_pairs/sgl/__init__.py'
patch=''.join(difflib.unified_diff(original.splitlines(True),(original+fragment).splitlines(True),fromfile='a/'+target,tofile='b/'+target))
target_module='TileOPs/benchmarks/ascend_all_pairs/sgl/integration700_sgl_conv_sampling02.py'
patch+=''.join(difflib.unified_diff([],module.read_text().splitlines(True),fromfile='/dev/null',tofile='b/'+target_module))
(R/'REGISTRATION.patch').write_text(patch)
# Execute exact registration statements with stubs; imports alone are removed.
tree=ast.parse(fragment);tree.body=[x for x in tree.body if not isinstance(x,ast.ImportFrom)]
old=lambda case:'OLD'; cv=lambda case:'CONV'; sa=lambda case:'SAMPLE'; env={'PROVIDERS':{'Conv2dFwdOp':old,'ChainSpeculativeSamplingFwdOp':old},'_i700_cs02_conv':cv,'_i700_cs02_chain':sa}
exec(compile(tree,'REGISTRATION_APPEND.py','exec'),env)
from types import SimpleNamespace
for row in rows: assert env['PROVIDERS'][row['operator']](SimpleNamespace(id=row['case_id']))==('CONV' if row['operator']=='Conv2dFwdOp' else 'SAMPLE')
for op in env['PROVIDERS']: assert env['PROVIDERS'][op](SimpleNamespace(id='out-of-scope'))=='OLD'
(R/'REGISTRATION_CPU_CHECK.json').write_text(json.dumps({'passed':True,'exact_keys':9,'outside_scope_fallback_preserved':True,'deployed':False},indent=2))
now=datetime.datetime.now(datetime.timezone.utc);start=datetime.datetime.fromisoformat(json.loads((R/'STARTED.json').read_text())['utc'])
guards=[]
for d,first in [('runs/attempt01','runs/attempt01'),('runs/attempt02','runs/attempt02'),('batch03','runs/attempt03'),('batch04','runs/attempt08'),('batch05','runs/attempt13')]+[(f'runs/attempt{i}',f'runs/attempt{i}') for i in range(18,23)]:
 g=json.loads((R/d/'GUARD_RESULT.json').read_text());m=json.loads((R/first/'output/GUARD_RUNNING.json').read_text());q=(datetime.datetime.fromisoformat(m['updated_at'])-datetime.datetime.fromisoformat(m['requested_at'])).total_seconds();guards.append({'receipt':str(R/d/'GUARD_RESULT.json'),'returncode':g['returncode'],'queue_seconds':q,'held_seconds':g['elapsed_seconds']-q})
result['updated_at']=now.isoformat(); (R/'RESULT.tmp').write_text(json.dumps(result,indent=2));(R/'RESULT.tmp').replace(R/'RESULT.json')
delivery={'finished_at':now.isoformat(),'deployment_host':socket.gethostname(),'status':'verified_unregistered','verified_keys':9,'checked_outputs':sum(x['checked_outputs'] for x in rows),'registered_keys':0,'rows':rows,'adapter_sha256':digest,'kernel_ownership':{'conv':'Original SGL libsgl_kernel_npu.so, torch.ops.npu.sgemmc_expand, csrc/lora/op_host and op_kernel sources hashed in checker','sampling':'Original sgl_kernel_npu/sample/chain_speculative_sampling.py complete API, hashed in checker'},'registration_patch':str(R/'REGISTRATION.patch'),'registration_append':str(R/'REGISTRATION_APPEND.py'),'review_backup':str(R/'provider_init.review_backup.py'),'deployment_instruction':'Parent only: within /tmp/baseline700-provider-sgl.lock, re-read latest shared __init__.py, make fresh backup, copy byte-identical adapter module, append exact registration fragment preserving all existing providers. Patch is review evidence, not permission to restore old snapshot. Run device-free exact-key check; no repeated numerical run needed for byte-identical migration.','reproduction':'python3 '+str(R/'launch.py')+' OP CASE NEW_ATTEMPT; launch.py invokes RUN_GUARDED.sh run_case.py sgl-kernel-npu OP CASE OUTPUT. Original baseline-run environment, no GDN custom OPP. Max 300s, owner baseline700-20261026/sgl_conv_sampling02.','remaining_issues':['Formal provider registration reserved to parent because worker writes are private-only.'],'environment_repair':str(R/'ENVIRONMENT_FIX.json'),'invalidated_batches':[str(R/'BATCH04_INVALIDATED.json'),str(R/'BATCH05_INVALIDATED.json')],'elapsed_seconds':(now-start).total_seconds(),'guard_timings':guards,'repeat_accounting':{'environment_failure_attempts':['attempt01','attempt02','attempt03'],'contaminated_attempts':['attempt08','attempt09','attempt10','attempt13','attempt14','attempt15','attempt16'],'unchanged_valid_successes_rerun':0},'changed_files':[str(p.relative_to(R)) for p in R.iterdir() if p.is_file() and p.name not in ['TASK.txt','SCOPE.json','RUN_GUARDED.sh']]}
(R/'DELIVERY.json').write_text(json.dumps(delivery,indent=2))
(R/'HEARTBEAT.json').write_text(json.dumps({'utc':now.isoformat(),'phase':'delivered_verified_unregistered','current_keys':[],'last_progress_at':now.isoformat(),'blocked_reason':'Private-only write boundary: parent registration required'},indent=2))
print(json.dumps({'keys':9,'outputs':delivery['checked_outputs'],'elapsed_seconds':delivery['elapsed_seconds'],'guard_timings':guards},indent=2))
