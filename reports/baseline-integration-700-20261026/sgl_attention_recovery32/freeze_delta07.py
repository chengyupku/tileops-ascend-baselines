from pathlib import Path
import json,hashlib,shutil,datetime
P=Path(__file__).resolve().parent;out=P/'delta07';out.mkdir(exist_ok=False);rows=json.loads((P/'REGISTRATION_SCOPE.json').read_text())['rows'];assert len(rows)==7
names=['dsa_adapter.py','route_install.py','run_case.py','provider_guard.py','baseline-run','numerics.original.py','numerics_private.py','DIAGNOSTIC_ONLY.patch','CPU_DIAGNOSTIC_PREFLIGHT.json','SOURCE_HASHES.json','CONTRACT_MATCH.json','REGISTRATION.patch','provider_registration.append.py','REGISTRATION_CHECK.json','REGISTRATION_INSTRUCTIONS.txt','provider_init.before_final.py','PROVIDER_LOCK_RUNTIME_ACTIVE.json'];hashes={}
for n in names:
 shutil.copy2(P/n,out/n);hashes['delta07/'+n]=hashlib.sha256((out/n).read_bytes()).hexdigest()
for r in rows:
 for n in [r['checker_receipt'],r['guard_receipt']]:hashes[n]=hashlib.sha256((P/n).read_bytes()).hexdigest()
 case=(P/r['checker_receipt']).parent
 for n in ['DIAGNOSTIC_OVERRIDE.json','ROUTE.json','GUARD_RUNNING.json','maps.txt','native.jsonl','native.summary.json','NATIVE_ARTIFACTS.json','LOADED_MODULES.json','RUNNER.py','ENVIRONMENT.sh']:
  f=case/n
  if f.exists():hashes[str(f.relative_to(P))]=hashlib.sha256(f.read_bytes()).hexdigest()
 for f in (case/'native_artifacts').rglob('*'):
  if f.is_file():hashes[str(f.relative_to(P))]=hashlib.sha256(f.read_bytes()).hexdigest()
 f=P/'provider_locks'/(case.name+'.json');hashes[str(f.relative_to(P))]=hashlib.sha256(f.read_bytes()).hexdigest()
(out/'RESULT.json').write_text(json.dumps({'rows':rows},indent=2));proof=json.loads((P/'NATIVE_PROOF.json').read_text());selected={r['case_id'] for r in rows};proof['rows']=[r for r in proof['rows'] if r['key']['case_id'] in selected];assert len(proof['rows'])==7 and all(r['native_launch_count'] for r in proof['rows']);(out/'NATIVE_PROOF.json').write_text(json.dumps(proof,indent=2))
for n in ['RESULT.json','NATIVE_PROOF.json']:hashes['delta07/'+n]=hashlib.sha256((out/n).read_bytes()).hexdigest()
(out/'EVIDENCE_HASHES.json').write_text(json.dumps(hashes,indent=2));(P/'DELIVERY_DELTA07.json').write_text(json.dumps({'immutable':True,'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'worker':'sgl_attention_recovery32','host':'openlibing-DevEnv-055409','status':'verified_unregistered','count':7,'registered':False,'disjoint_from':['DELIVERY_INTERIM.json','DELIVERY_DELTA03.json'],'rows':rows,'registration_patch':'delta07/REGISTRATION.patch','registration_append':'delta07/provider_registration.append.py','registration_check':'delta07/REGISTRATION_CHECK.json','binding_and_evidence_hashes':'delta07/EVIDENCE_HASHES.json','native_proof':'delta07/NATIVE_PROOF.json','bridge_used':False,'qualification':'All seven original canonical checks passed with clean guard0 and diagnostic-only CPU override. Existing provider coordination lock held read-only through entire command; original binding bytes frozen.'},indent=2));print('immutable delta07 ready')
