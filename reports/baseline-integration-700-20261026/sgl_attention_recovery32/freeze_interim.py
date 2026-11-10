from pathlib import Path
import json,hashlib,shutil,datetime
p=Path(__file__).resolve().parent;out=p/'interim14';out.mkdir(exist_ok=True);assert not (p/'DELIVERY_INTERIM.json').exists()
keys={tuple(k[x] for x in ('baseline','operator','case_id')) for k in json.loads((p/'RECOMMENDED_ACCEPT.json').read_text())['keys']}
rows=[r for r in json.loads((p/'RESULT.json').read_text())['rows'] if tuple(r[x] for x in ('baseline','operator','case_id')) in keys];assert len(rows)==14
names=['attention_adapter.py','attention_page16.py','dsa_adapter.py','integration700_sgl_attention32.py','prefill_adapter.py','cache_adapter.py','route_install.py','run_case.py','baseline-run','numerics.original.py','numerics_private.py','DIAGNOSTIC_ONLY.patch','CPU_DIAGNOSTIC_PREFLIGHT.json','REGISTRATION.patch','provider_registration.append.py','REGISTRATION_CHECK.json','REGISTRATION_INSTRUCTIONS.txt','SOURCE_HASHES.json','CONTRACT_MATCH.json','NATIVE_PROOF.json','BYTECODE_NOTE.json']
hashes={}
for n in names:
 shutil.copy2(p/n,out/n);hashes['interim14/'+n]=hashlib.sha256((out/n).read_bytes()).hexdigest()
for r in rows:
 for n in [r['checker_receipt'],r['guard_receipt']]:hashes[n]=hashlib.sha256((p/n).read_bytes()).hexdigest()
 case=(p/r['checker_receipt']).parent
 for n in ['DIAGNOSTIC_OVERRIDE.json','ROUTE.json','GUARD_RUNNING.json','maps.txt','native.jsonl','native.summary.json','NATIVE_ARTIFACTS.json','LOADED_MODULES.json']:
  f=case/n
  if f.exists():hashes[str(f.relative_to(p))]=hashlib.sha256(f.read_bytes()).hexdigest()
 for f in (case/'native_artifacts').rglob('*'):
  if f.is_file():hashes[str(f.relative_to(p))]=hashlib.sha256(f.read_bytes()).hexdigest()
for n in ['REPS02_SGL_NATIVE.json','REPS03_SGL_NATIVE.json']:
 shutil.copy2(p/n,out/n);hashes['interim14/'+n]=hashlib.sha256((out/n).read_bytes()).hexdigest()
(out/'EVIDENCE_HASHES.json').write_text(json.dumps(hashes,indent=2));(out/'RESULT.json').write_text(json.dumps({'rows':rows},indent=2))
(p/'DELIVERY_INTERIM.json').write_text(json.dumps({'immutable':True,'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'worker':'sgl_attention_recovery32','host':'openlibing-DevEnv-055409','status':'verified_unregistered','count':14,'registered':False,'rows':rows,'registration_patch':'interim14/REGISTRATION.patch','registration_append':'interim14/provider_registration.append.py','registration_check':'interim14/REGISTRATION_CHECK.json','binding_and_evidence_hashes':'interim14/EVIDENCE_HASHES.json','numerics_original_sha256':hashlib.sha256((out/'numerics.original.py').read_bytes()).hexdigest(),'numerics_private_sha256':hashlib.sha256((out/'numerics_private.py').read_bytes()).hexdigest(),'diagnostic_only_diff':'interim14/DIAGNOSTIC_ONLY.patch','native_proof':'interim14/NATIVE_PROOF.json','bridge_used':False,'qualification':json.loads((p/'RECOMMENDED_ACCEPT.json').read_text())['qualification'],'remaining_work':'Remaining DSA expansion continues separately. These successful keys will not be numerically rerun.','shared_bytecode_note':'interim14/BYTECODE_NOTE.json'},indent=2));print('frozen14; hashed files',len(hashes))
