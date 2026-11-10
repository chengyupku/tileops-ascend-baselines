import json,hashlib,datetime
from pathlib import Path
W=Path(__file__).resolve().parent
rows=json.loads((W/'RESULT.json').read_text());contracts=json.loads((W/'CONTRACTS.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
audit=[]
for row in rows:
 c=row['exactkey'][2];d=W/c
 if not (d/'checker.json').is_file():continue
 x=json.loads((d/'checker.json').read_text());g=json.loads((d/'GUARD_RECEIPT.json').read_text());running=json.loads((d/'GUARD_RUNNING.json').read_text());libs=json.loads((d/'loaded-libraries.json').read_text()) if (d/'loaded-libraries.json').exists() else []
 recipe=json.loads((W/'chunk_package/RUNTIME_RECIPE.json').read_text()); libmap={v['path']:v['sha256'] for v in libs};pins={v['path']:v['sha256'] for v in json.loads((W/'chunk_package/RUNTIME_PINS.json').read_text())['files']}
 checks={'all_outputs':x.get('verification',{}).get('checked_outputs')==x.get('verification',{}).get('total_outputs')==2,'inputs_preserved':x.get('inputs_preserved') is True,'seed_recorded':isinstance(x.get('input_seed'),int),'params_match_original_contract':x.get('case_snapshot',{}).get('parameters')==contracts[c]['params'],'no_source_change':x.get('provider_source_changed') is False and x.get('oracle_source_changed') is False,'no_sync_cleanup_error':'sync_error' not in x and 'cleanup_error' not in x,'guard_success':g['returncode']==0,'owner_match':running['owner']=='baseline700-20261026/gdn_capacity_chunk16','loaded_original_opapi':libmap.get(recipe['opapi'])==pins[recipe['opapi']],'loaded_original_nnopbase':libmap.get(json.loads((W/'RUNTIME_REALPATHS.json').read_text())['nnopbase'])==pins[recipe['nnopbase']],'verified':x.get('status')=='verified'}
 audit.append({'exactkey':row['exactkey'],'checks':checks,'all_checks_passed':all(checks.values()),'seed':x.get('input_seed'),'verification':x.get('verification'),'queue_seconds':g['queue_seconds'],'child_seconds':g['child_seconds'],'sync_cleanup_evidence':'Original checker finally synchronizes and invokes candidate.cleanup; DirectACL cleanup synchronizes/destroys descriptors. Absence of error is evidence of normal completion, not separately instrumented events.','evidence_sha256':{p.name:sha(p) for p in d.iterdir() if p.is_file()}})
(W/'DELIVERY_AUDIT.json').write_text(json.dumps(audit,indent=2))
print(json.dumps({'receipts':len(audit),'all_checks_passed':sum(x['all_checks_passed'] for x in audit),'failures':[{ 'key':x['exactkey'],'checks':x['checks']} for x in audit if not x['all_checks_passed']]}))
