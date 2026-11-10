from pathlib import Path
import json,hashlib,datetime,subprocess
W=Path(__file__).resolve().parent;case='q35-gva-decode-b512-bfloat16';d=W/case
c=json.loads((d/'checker.json').read_text());m=json.loads((d/'guard_metadata.json').read_text());g=json.loads((d/'GUARD_RECEIPT.json').read_text());parse=datetime.datetime.fromisoformat
assert c['status']=='verified' and c['verification']['checked_outputs']==2 and c['inputs_preserved'] and g['returncode']==0
assert c['case_snapshot']['parameters']==json.loads((W/'CONTRACTS.json').read_text())[case]['params']
assert not c['provider_source_changed'] and not c['oracle_source_changed']
for n,h in json.loads((W/'FREEZE.json').read_text()).items():assert hashlib.sha256((W/n).read_bytes()).hexdigest()==h
libs=json.loads((d/'loaded-libraries.json').read_text());r=json.loads((W/'recurrent_package/RUNTIME_RECIPE.json').read_text());actual={v['path']:v['sha256'] for v in libs}
source=c['source_sha256'];selected={p:h for p,h in source.items() if 'recurrent_kda' in p or p in [r['opapi'],r['nnopbase']]}
(W/'NATIVE_EVIDENCE.json').write_text(json.dumps({'source_artifact_sha256':selected,'loaded_original_opapi':{p:h for p,h in actual.items() if 'libcust_opapi.so' in p or 'libnnopbase.so' in p},'binding_native_call_line':33,'transport_native_load_lines':[32,33],'transport_native_launch_lines':[98,99],'transport_sync_cleanup_lines':[106,107,111],'checker_seed_line':241,'checker_inputs_preserved_line':391,'checker_sync_cleanup_lines':[408,412,414],'full_loaded_libraries':str(d/'loaded-libraries.json'),'full_source_hashes':str(d/'checker.json')},indent=2))
summary={k:c[k] for k in ['status','api','input_seed','verification','inputs_preserved','provider_source_changed','oracle_source_changed','wall_seconds','verifier_sha256','limitations','reference_device','independent_cpu_reference']}
summary['queue_seconds']=(parse(m['acquired_at'])-parse(m['requested_at'])).total_seconds();summary['request_to_child_seconds']=(parse(m['updated_at'])-parse(m['requested_at'])).total_seconds();summary['guard_total_seconds']=g['elapsed_seconds'];summary['guard_returncode']=g['returncode']
(W/'CHECK_SUMMARY.json').write_text(json.dumps(summary,indent=2))
result=[{'key':['ops-transformer','GDNFwdOp',case],'baseline':'ops-transformer','operator':'GDNFwdOp','case_id':case,'status':'verified_pending_parent_acceptance','checker_status':'verified','checked_outputs':2,'total_outputs':2,'registered':False,'accepted_added':0,'evidence':{'checker':str(d/'checker.json'),'guard':str(d/'GUARD_RECEIPT.json'),'metadata':str(d/'guard_metadata.json'),'summary':str(W/'CHECK_SUMMARY.json'),'native':str(W/'NATIVE_EVIDENCE.json'),'diff':str(W/'adapter.diff')},'remaining':['Parent review and serial provider registration','No explicit free-HBM measurement was added; original conservative device envelope retained.'],'timings':summary}]
(W/'RESULT.json').write_text(json.dumps(result,indent=2));(W/'HEARTBEAT.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'case':case,'phase':'complete_verified_pending_parent','accepted_added':0},indent=2))
# A finite delivery sync, after device exit; only private directory files.
files=[p for p in W.iterdir() if p.is_file() and p.name not in ['SPAWN.json','TASK.txt','CONTRACTS.json','CASE_RESOURCE_ESTIMATES.json']]
code='from pathlib import Path\n'
for p in files:code+='Path('+repr(str(p))+').write_text('+repr(p.read_text())+')\n'
p=subprocess.run(['/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh','openlibing-DevEnv-055409','python3 -S -B -'],input=code,text=True,capture_output=True,check=True)
print(json.dumps(summary))
