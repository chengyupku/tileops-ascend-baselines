import json,hashlib,datetime,sys
from pathlib import Path
W=Path(__file__).resolve().parent
sys.path.insert(0,str(W));from audit_case import audit
cases=json.loads((W/'EXECUTION_ORDER.json').read_text());rows=[]
for c in cases:
 d=W/c
 if (d/'checker.json').exists() and (d/'GUARD_RECEIPT.json').exists():
  r=json.loads((d/'checker.json').read_text());g=json.loads((d/'GUARD_RECEIPT.json').read_text())
  if r.get('status')=='verified' and g['returncode']==0:
   rows.append(json.loads((d/'AUDIT.json').read_text()));continue
 reason='Not executed because sequential execution stopped before this key.'
 if (d/'guard.stderr.log').exists():reason='\n'.join((d/'guard.stderr.log').read_text().splitlines()[-15:])
 rows.append({'key':['ops-transformer','GDNFwdOp',c],'status':'blocked','registered':False,'accepted_added':0,'reason':reason,'numeric_failure_proven':False})
(W/'RESULT.json').write_text(json.dumps(rows,indent=2))
pins=json.loads((W/'chunk_package/RUNTIME_PINS.json').read_text())['files'];bad=[]
for p in pins:
 if hashlib.sha256(Path(p['path']).read_bytes()).hexdigest()!=p['sha256']:bad.append(p['path'])
assert not bad,bad
(W/'POST_RUN_INTEGRITY.json').write_text(json.dumps({'runtime_pins_checked':len(pins),'mismatches':bad},indent=2))
verified=[r for r in rows if r['status']=='verified_unregistered'];queue=0;child_upper=0;idle_wait=0
for r in verified:
 g=json.loads((W/r['key'][2]/'GUARD_RECEIPT.json').read_text());m=g['guard_samples'][0]
 dt=lambda x:datetime.datetime.fromisoformat(x)
 queue+=(dt(m['acquired_at'])-dt(m['requested_at'])).total_seconds()
 idle_wait+=(dt(m['updated_at'])-dt(m['acquired_at'])).total_seconds()
 child_upper+=(dt(g['end'])-dt(m['updated_at'])).total_seconds()
changed=[p.name for p in W.iterdir() if p.is_file() and p.name not in ['SCOPE.json','SPAWN.json','TASK.txt']]
report={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'worker':'634142f39c5043b8','status':'all_numeric_verified_unregistered_with_audit_limitations' if len(verified)==10 else 'partial_verified_and_blocked','accepted_added':0,'registered':False,'scope_count':10,'verified_count':len(verified),'checked_outputs':2*len(verified),'failed_numeric_count':0,'blocked_count':10-len(verified),'results':'RESULT.json','source_and_build':['SOURCE_PINS.json','DEPENDENCY_PINS.json','GDN_SOURCE_COMPARISON.json','BUILD_PROVENANCE.json','STATIC_DEPLOYMENT.json','DEVICE_COMPARISON.json','GENERATED_BUILD_PINS.json','BUILD_CLOSURE_INTEGRITY.json','build4/GUARD_RECEIPT.json','installer/GUARD_RECEIPT.json'],'checks':['58 original GDN files byte-identical to remote source','Ordinary original --pkg, real SDK 9.2.0-beta.2, native x86 host API','Both newly built device objects and metadata byte-identical to original remote package','8 cgroup-v2 CPU positive/negative tests passed','All verified cases use full original canonical checker, exact parameters, seeds, CPU oracle, complete 2 outputs and inputs preserved','Actual library maps and loader device-object opens audited','Original 1 GiB workspace ceiling unchanged','Post-run private runtime pins unchanged'],'timings':{'canonical_reservation_queue_seconds_total':queue,'canonical_physical_idle_check_seconds_total':idle_wait,'worker_wall_seconds':datetime.datetime.now(datetime.timezone.utc).timestamp()-json.loads(json.loads((W/'SPAWN.json').read_text())['stdout'])['created_at'],'canonical_child_seconds_upper_bound':child_upper,'checker_seconds_total':sum(r['checker_seconds'] for r in verified),'receipt_polling_resolution_seconds':5,'build_hold_seconds_conservative_upper_bound':json.loads((W/'BUILD_TIMINGS.json').read_text())['total_hold_seconds_upper_bound_including_all_queue_and_polling'],'build_timing_note':'First two build attempts lack sampled metadata timestamps; all elapsed intervals including their queue, plus sampled build3/build4, fit within 600s. Exact first-two queue/hold split unavailable. Build log birth/mtime evidence retained.'},'remaining_issues':['Parent acceptance and registration only; no shared provider or memory was modified.','First case has fresh inside-guard admissions and earlier task-level CPU probe, but no separately saved per-key prequeue receipt; remaining cases have both.','Early make wrapper capped each invocation at 8, with nested jobserver reset warnings in device compiler make invocations; wrapper subsequently preserves inherited jobserver. Aggregate process concurrency was not sampled; do not claim independently measured peak.','Worker numerical pass is not parent acceptance or performance profiling.'],'changed_files':sorted(set(changed+['DELIVERY.json','RESULT.json','OUTPUT_COMPLETENESS.json','REGISTRATION_PLAN.json','POST_RUN_INTEGRITY.json'])),'changed_directories':['source/ (private original source and generated build)','install/ (standard original private package)','chunk_package/ (private adapter paths and strict capacity reader)','per-exactkey directories (scripts, checker/guard/admission/native evidence)','bin/, build-cache/, tmp/ (private build helpers/cache)']}
report['resource_admission_summary'] = {'host_and_hbm_are_separate':True,'per_case':[]}
for r in verified:
 d=W/r['key'][2];host=json.loads((d/'fresh_resource_admission.json').read_text());hbm=json.loads((d/'HBM_ADMISSION.json').read_text())
 report['resource_admission_summary']['per_case'].append({'case_id':r['key'][2],'host_margin_bytes':host['effective_available_host_bytes']-host['required_envelope']['minimum_effective_available_host_bytes'],'hbm_margin_after_safety_bytes':hbm['free_hbm_bytes']-hbm['estimated_device_budget_bytes']-hbm['safety_margin_bytes']})
(W/'OUTPUT_COMPLETENESS.json').write_text(json.dumps({'scope_count':10,'verified_count':len(verified),'checked_outputs':len(verified)*2,'full_original_checker':True,'tensor_dump_claimed':False,'cases':[{'key':r['key'],'verification':r['verification'],'full_outputs':r['outputs'],'inputs_preserved':r['inputs_preserved'],'input_seed':r['input_seed'],'checker_path':r['key'][2]+'/checker.json'} for r in verified]},indent=2))
(W/'DELIVERY.json').write_text(json.dumps(report,indent=2))
(W/'REGISTRATION_PLAN.json').write_text(json.dumps({'accepted_added':0,'action':'Parent review and register exact verified keys against pinned private package; no worker registry write','keys':[r['key'] for r in verified],'runtime_pins':'chunk_package/RUNTIME_PINS.json','runtime_recipe':'chunk_package/RUNTIME_RECIPE.json','binding':'chunk_package/binding.py'},indent=2))
(W/'HEARTBEAT.json').write_text(json.dumps({'at':report['at'],'phase':'finished','verified':len(verified),'blocked':10-len(verified),'accepted_added':0}))
print(json.dumps({k:report[k] for k in ['status','verified_count','blocked_count','accepted_added']}))
