from pathlib import Path
import json,hashlib,shutil,datetime,difflib
P=Path(__file__).resolve().parent;out=P/'delta03';out.mkdir(exist_ok=True);assert not (P/'DELIVERY_DELTA03.json').exists()
old={tuple(r[k] for k in ('baseline','operator','case_id')) for r in json.loads((P/'DELIVERY_INTERIM.json').read_text())['rows']};rows=[r for r in json.loads((P/'RESULT.json').read_text())['rows'] if r['status']=='verified_unregistered' and tuple(r[k] for k in ('baseline','operator','case_id')) not in old];rows.sort(key=lambda r:json.loads((P/r['checker_receipt']).read_text())['finished_at']);rows=rows[:3];assert len(rows)==3
names=['dsa_adapter.py','route_install.py','run_case.py','baseline-run','numerics.original.py','numerics_private.py','DIAGNOSTIC_ONLY.patch','CPU_DIAGNOSTIC_PREFLIGHT.json','SOURCE_HASHES.json','CONTRACT_MATCH.json'];hashes={}
for n in names:
 shutil.copy2(P/n,out/n);hashes['delta03/'+n]=hashlib.sha256((out/n).read_bytes()).hexdigest()
for r in rows:
 for n in [r['checker_receipt'],r['guard_receipt']]:hashes[n]=hashlib.sha256((P/n).read_bytes()).hexdigest()
 case=(P/r['checker_receipt']).parent
 for n in ['DIAGNOSTIC_OVERRIDE.json','ROUTE.json','GUARD_RUNNING.json','maps.txt','native.jsonl','native.summary.json','NATIVE_ARTIFACTS.json','LOADED_MODULES.json','RUNNER.py','ENVIRONMENT.sh']:
  f=case/n
  if f.exists():hashes[str(f.relative_to(P))]=hashlib.sha256(f.read_bytes()).hexdigest()
 for f in (case/'native_artifacts').rglob('*'):
  if f.is_file():hashes[str(f.relative_to(P))]=hashlib.sha256(f.read_bytes()).hexdigest()
text=(P/'interim14/provider_registration.append.py').read_text();text='\n'.join('_ar32_keys = '+repr({'DSADecodeWithKVCacheFwdOp':[r['case_id'] for r in rows]}) if l.startswith('_ar32_keys = ') else l for l in text.splitlines())+'\n';text=text.replace('_ar32','_ar32_delta03');(out/'provider_registration.append.py').write_text(text)
before=(out/'provider_init.before.py').read_text();(out/'REGISTRATION.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True),(before+text).splitlines(True),fromfile='a/TileOPs/benchmarks/ascend_all_pairs/sgl/__init__.py',tofile='b/TileOPs/benchmarks/ascend_all_pairs/sgl/__init__.py')))
(out/'RESULT.json').write_text(json.dumps({'rows':rows},indent=2));proof=json.loads((P/'NATIVE_PROOF.json').read_text());selected={r['case_id'] for r in rows};proof['rows']=[r for r in proof['rows'] if r['key']['case_id'] in selected];assert len(proof['rows'])==3 and all(r['native_launch_count'] for r in proof['rows']);(out/'NATIVE_PROOF.json').write_text(json.dumps(proof,indent=2))
for n in ['provider_registration.append.py','REGISTRATION.patch','RESULT.json','NATIVE_PROOF.json']:hashes['delta03/'+n]=hashlib.sha256((out/n).read_bytes()).hexdigest()
(out/'EVIDENCE_HASHES.json').write_text(json.dumps(hashes,indent=2));check=json.loads((out/'REGISTRATION_CHECK.json').read_text()) if (out/'REGISTRATION_CHECK.json').exists() else None
(out/'delivery.pending.json').write_text(json.dumps({'immutable':True,'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'worker':'sgl_attention_recovery32','host':'openlibing-DevEnv-055409','status':'verified_unregistered','count':3,'registered':False,'disjoint_from':'DELIVERY_INTERIM.json','rows':rows,'registration_patch':'delta03/REGISTRATION.patch','registration_append':'delta03/provider_registration.append.py','registration_check':'delta03/REGISTRATION_CHECK.json','binding_and_evidence_hashes':'delta03/EVIDENCE_HASHES.json','native_proof':'delta03/NATIVE_PROOF.json','bridge_used':False,'qualification':'All three original canonical checks passed with clean guard0 and diagnostic-only CPU override. Exact original binding bytes preserved.'},indent=2));print('immutable delta03 ready')
