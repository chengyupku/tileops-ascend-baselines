from pathlib import Path
import json,hashlib,datetime,difflib,re
R=Path(__file__).resolve().parent;W=Path('/root/dyq/workspace')
rows=[]
for s in json.loads((R.parent/'SCOPE.json').read_text()):
 op=s['operator'];row={'key':s['key'],'changed_files':[str(R/'run_case.py'),str(R/'launch.py')],'guard_owner':'selected-integration-20261026/nn_misc_math_recovery01','state_cache_outputs':'No final_state/cache output in this canonical contract; see checked_outputs/total_outputs for actual verification.'}
 if op=='SoftmaxFwdOp':
  h=json.loads((R/'historical_softmax/SOFTMAX_GUARD_RECOVERY.json').read_text());d=json.loads(Path(h['checker']['path']).read_text())
  row.update(status='skipped_already_completed',api=d['api'],checked_outputs=d['verification']['checked_outputs'],total_outputs=d['verification']['total_outputs'],guard_exit=h['guard_exit'],checker=h['checker']['path'],guard=h['guard_exit_file']['path'],reason='Exact original historical full checker and guard exit 0 authenticated; no successful-case replay.',new_execution=False)
 else:
  run=R/'runs'/op;out=run/'output';d=json.loads((out/'checker.json').read_text()) if (out/'checker.json').exists() else {};g=json.loads((run/'GUARD_RESULT.json').read_text()) if (run/'GUARD_RESULT.json').exists() else {}
  v=d.get('verification',{});native=[json.loads(l) for l in (out/'native.jsonl').read_text().splitlines()] if (out/'native.jsonl').exists() else []
  needle={'BatchNormBwdOp':'batchnormgradv3','CountNonzeroFwdOp':'lpnormv2','VectorNormFwdOp':'lpnormv2','SumFwdOp':'reducesum'}[op]
  hits=[x for x in native if needle in x['text'].lower()];comp=[x for x in hits if 'single op compile success' in x['text']];launch=[x for x in hits if 'StreamLaunchKernel' in x['text']]
  evidence={'api':needle,'compile_success':comp,'native_launch':launch,'all_matching_events':hits}
  (out/'ORIGINAL_API_EVIDENCE.json').write_text(json.dumps(evidence,indent=2))
  passed=g.get('returncode')==0 and d.get('status')=='verified' and v.get('checked_outputs',0)>0 and v['checked_outputs']==v['total_outputs'] and bool(comp) and bool(launch)
  row.update(status='verified' if passed else 'blocked',api=d.get('api',needle),guard_exit=g.get('returncode'),checker=str(out/'checker.json'),guard=str(run/'GUARD_RESULT.json'),native_evidence=str(out/'ORIGINAL_API_EVIDENCE.json'),checked_outputs=v.get('checked_outputs',0),total_outputs=v.get('total_outputs',3 if op=='BatchNormBwdOp' else 1),reason=None if passed else d.get('error','No complete numerical/native/guard success'),new_execution=True,inputs_preserved=d.get('inputs_preserved'),provider_source_changed=d.get('provider_source_changed'),oracle_source_changed=d.get('oracle_source_changed'))
  if op=='SumFwdOp':row['changed_files'].append(str(R/'sum_adapter.py'))
 rows.append(row)
result={'scope_count':5,'host':'local','finished_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rows':rows,'summary':{x:sum(r['status']==x for r in rows) for x in ['verified','skipped_already_completed','blocked']}}
(R/'RESULT.json').write_text(json.dumps(result,indent=2))
(R/'launcher.patch').write_text(''.join(difflib.unified_diff((R.parent/'private/run_case.py').read_text().splitlines(True),(R/'run_case.py').read_text().splitlines(True),fromfile='private/run_case.py',tofile='recovery01/run_case.py')))
files=[R/f for f in ['run_case.py','launch.py','sum_adapter.py','native_log_capture.py']]
files+=list((R/'providers').rglob('*.py'))
(R/'DELIVERY_PINS.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
(R/'DELIVERY.json').write_text(json.dumps({'status':result['summary'],'callable_entry':str(R/'run_case.py'),'launcher':str(R/'launch.py'),'run_method':'python3 '+str(R/'launch.py')+' OPERATOR (fresh runs/OPERATOR required; never replay a completed key)','guard_command_template':"npu-guard run --owner 'selected-integration-20261026/nn_misc_math_recovery01' --wait 3600 --timeout 300 -- /root/dyq/workspace/.tools/baselines/baseline-run BASELINE python "+str(R/'run_case.py')+' BASELINE OPERATOR CASE_ID FRESH_OUTPUT_DIR','patch':str(R/'launcher.patch'),'adapters':{'BatchNormBwdOp':'providers/ops_nn/normalization.py::normalization','CountNonzeroFwdOp':'providers/ops_nn/normalization.py::count_nonzero','VectorNormFwdOp':'providers/ops_nn/normalization.py::normalization','SumFwdOp':'sum_adapter.py::make','SoftmaxFwdOp':'historical_softmax/ (preserved, no rerun)'},'checker':'/root/dyq/workspace/TileOPs/scripts/ascend_pair_check.py (unaltered shared checker, pinned CHECKER_PIN.json)','integration':'Provider overlay precedes only benchmarks.ascend_all_pairs package path; exact requested provider is asserted private and registered in PROVIDERS. Shared provider/registry/vendor/compiler/oracle remain unchanged. Acceptance JSON is not used as validation proof.','unresolved':[r for r in rows if r['status']=='blocked'],'result':str(R/'RESULT.json')},indent=2))
print(json.dumps(result['summary']))
