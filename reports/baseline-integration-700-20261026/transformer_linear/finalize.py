from pathlib import Path
import json,hashlib,datetime,subprocess,sys,collections
W=Path(__file__).resolve().parent
for n in ['update_result.py','freeze.py','static_registration_check.py','timing_audit.py','make_delivery.py']:subprocess.run([sys.executable,'-B',str(W/n)],check=True)
r=json.loads((W/'RESULT.json').read_text());s=json.loads((W/'SCOPE.json').read_text());keys={(x['baseline'],x['operator'],x['case_id']) for x in s};seen=set();passed=[]
for x in r['rows']:
 k=(x['baseline'],x['operator'],x['case_id']);assert k in keys and k not in seen;seen.add(k)
 if x['status']=='completed':
  assert x['registered'] and x['checked_outputs']==x['total_outputs']==2
  c=json.loads((W/x['checker_receipt']['path']).read_text());assert c['status']=='verified' and c['inputs_preserved'] and not c['oracle_source_changed'] and not c['provider_source_changed']
  for n in ['checker_receipt','guard_receipt']:
   f=W/x[n]['path'];assert hashlib.sha256(f.read_bytes()).hexdigest()==x[n]['sha256']
  passed.append(k)
assert seen==keys,keys-seen
check=dict(scope_count=len(keys),row_count=len(seen),completed_count=len(passed),unique_exact_keys=True,all_completed_full_outputs_and_inputs_preserved=True,receipt_hashes_match=True,only_accepted_statuses=all(x['status'] in ['completed','verified_unregistered','already_completed','blocked','failed'] for x in r['rows']),numerical_repeats_of_completed_keys=0)
(W/'FINAL_CHECKS.json').write_text(json.dumps(check,indent=2))
now=datetime.datetime.now(datetime.timezone.utc).isoformat();(W/'HEARTBEAT.json').write_text(json.dumps(dict(utc=now,phase='finished',current_keys=[],last_progress_at=now,blocked_reason='All scope keys have verified registration or explicit recorded failure/blocker',counts=r['counts']),indent=2))
artifacts=[]
for f in W.rglob('*'):
 if not f.is_file() or f.name in ('ARTIFACTS.json','DELIVERY.json') or any(part.startswith(('TMPDIR','XDG','ASCEND_','TORCH_','TRITON_','PIP_','CCACHE','__pycache__')) for part in f.relative_to(W).parts):continue
 artifacts.append(dict(path=str(f.relative_to(W)),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
(W/'ARTIFACTS.json').write_text(json.dumps(artifacts,indent=2))
d=json.loads((W/'DELIVERY.json').read_text());d['status']='scope_exhausted';d['final_checks']='FINAL_CHECKS.json';d['private_artifacts_manifest']='ARTIFACTS.json';d['timing_audit']='TIMING_AUDIT.json';d['route_boundary_audit']='ROUTE_AUDIT.json';d['source_changed_case']='q35-397b-tp8-1024-7168-bfloat16: numerical2/2 but unrelated attention modules changed shared provider snapshot; not counted and not re-run.'
d['private_changed_scripts']=[x['path'] for x in artifacts if x['path'].endswith('.py')]
(W/'DELIVERY.json').write_text(json.dumps(d,indent=2));print('FINAL',r['counts'])
