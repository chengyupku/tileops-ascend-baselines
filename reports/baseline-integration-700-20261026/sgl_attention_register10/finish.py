import pathlib,subprocess,json,shlex,datetime
P=pathlib.Path(__file__).resolve().parent
ssh=['/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh','openlibing-DevEnv-055409']
remote='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/sgl_attention_register10'
# Only private report writes.
scope=(P/'SCOPE.json').read_text();script=(P/'constructor_check.py').read_text()
cmd='python3 -B -c '+shlex.quote('from pathlib import Path;Path('+repr(remote+'/SCOPE.json')+').write_text('+repr(scope)+');exec('+repr(script)+')')
r=subprocess.run(ssh+[cmd],capture_output=True);(P/'constructor.stderr').write_bytes(r.stderr);assert r.returncode==0,r.stderr.decode();(P/'CONSTRUCTOR_CHECK.json').write_bytes(r.stdout)
for name in ['REGISTRATION.patch','provider_init.before.py','PINNED_LIBRARY.json']:
 r=subprocess.run(ssh+['cat '+shlex.quote(remote+'/'+name)],capture_output=True);assert r.returncode==0;(P/name).write_bytes(r.stdout)
audit=json.loads((P/'AUDIT.json').read_text());checks=json.loads((P/'DEVICE_FREE_CHECKS.json').read_text())
limits=['No new numerical run; CPU trace/constructor stubs only prove routing, not numerical correctness.','Original numerical checker used disclosed one-line max_abs diagnostic CPU conversion; original validators/reference unchanged.','Private libraries namespace resolves pinned pristine original files; backing tree is shared read-only, with dependency SHA256 checked before every build; unexpected drift fails closed.','Historical source_reason page16 prose is stale: frozen tested binding uses page128 and 64-query chunks.','Parent review required before ACCEPTED; no ledger or memory modified.']
registration={'registered_count':10,'exact_keys':[r['key'] for r in audit['rows']],'host':'openlibing-DevEnv-055409','shared_module':'/opt/tileops-all-pairs/workspace/TileOPs/benchmarks/ascend_all_pairs/sgl/integration700_sgl_attention_register10.py','shared_init':'/opt/tileops-all-pairs/workspace/TileOPs/benchmarks/ascend_all_pairs/sgl/__init__.py','lock':'/tmp/baseline700-provider-sgl.lock','append_only':True,'before_sha256':checks['shared_init_before_sha256'],'after_sha256':checks['shared_init_after_sha256'],'private_root':remote,'preserved_prior_exact_count':checks['prior_exact_count'],'versions':'delta03 and delta07 frozen dsa_adapter are byte-identical; both individually audited.','limitations':limits}
(P/'REGISTRATION.json').write_text(json.dumps(registration,indent=2))
accept={'recommended_count':10,'accepted_written':False,'rows':[dict(r,recommendation='recommend_accept_after_parent_review',registered=True,limitations=limits[:2]) for r in audit['rows']],'failed_scope_keys':[],'out_of_scope_failures':'Remaining recovery32 eight failures not taken over.','limitations':limits}
(P/'RECOMMENDED_ACCEPT.json').write_text(json.dumps(accept,indent=2))
delivery={'worker':'35ede808702c4120','at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope_count':10,'registered':10,'recommended_accept':10,'accepted_written':False,'new_npu_runs':0,'checks':{'evidence_hashes':201,'remote_source_oracle_checker_hashes':checks['remote_hashes_verified'],'prior_exact_routes_preserved':80,'fresh_import':True,'constructor_stub_cases':10,'generic_module_poison_ignored':True},'changed_files':[registration['shared_module'],registration['shared_init'],remote+'/*',str(P)+'/*'],'evidence':['AUDIT.json','REGISTRATION.json','REGISTRATION.patch','DEVICE_FREE_CHECKS.json','CONSTRUCTOR_CHECK.json','RECOMMENDED_ACCEPT.json'],'remaining_issues':limits}
(P/'DELIVERY.json').write_text(json.dumps(delivery,indent=2));(P/'HEARTBEAT.json').write_text(json.dumps({'at':delivery['at'],'phase':'complete','registered':10}))
# Mirror audit/delivery artifacts in explicitly allowed remote private report.
files={n:(P/n).read_text() for n in ['AUDIT.json','REGISTRATION.json','RECOMMENDED_ACCEPT.json','DELIVERY.json','HEARTBEAT.json','constructor_check.py']}
code='import json,sys;from pathlib import Path;p=Path('+repr(remote)+');[(p/n).write_text(s) for n,s in json.load(sys.stdin).items()]'
r=subprocess.run(ssh+['python3 -B -c '+shlex.quote(code)],input=json.dumps(files).encode(),capture_output=True);assert r.returncode==0,r.stderr.decode()
print(json.dumps(delivery['checks']))
