from pathlib import Path
import json,hashlib,datetime,collections,ast
W=Path(__file__).resolve().parent;R=W.parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads((W/'RESULT.json').read_text());reg=json.loads((W/'REGISTRATION.json').read_text());success=[x for x in r['rows'] if x['status']=='completed'];mappings=[]
for row in success:
 i=row['case_id'];child=W/i/'child.sh';formal=child.with_name('formal_child.sh');formal.write_text(child.read_text().replace(str(W/'chunk_package/entry.py'),str(W/'launch_formal.py')))
 c=json.loads((W/i/'checker.json').read_text());mappings.append(dict(key=[row['baseline'],row['operator'],i],factory=row['factory'],checker=row['checker_receipt'],source_sha256=c['source_sha256'],recipe=['npu-guard','run','--owner','baseline700-20261026/transformer_linear','--wait','3600','--timeout','300','--','bash',str(formal)]))
files=['TileOPs/benchmarks/ascend_all_pairs/ops_transformer/integration700_transformer_linear_gdn.py','TileOPs/benchmarks/ascend_all_pairs/ops_transformer/__init__.py']
start=json.loads((W/'STARTED.json').read_text())['utc'];now=datetime.datetime.now(datetime.timezone.utc)
receipts=[]
for f in W.glob('batch*/GUARD_RECEIPT.json'):receipts.append(json.loads(f.read_text()))
for i in ['q35-397b-tp8-2048-bfloat16','mamba2-2.7b-decode-b1-bfloat16','mamba2-780m-float16-float32']:
 f=W/i/'GUARD_RECEIPT.json'
 if f.exists():receipts.append(json.loads(f.read_text()))
d=dict(updated_at=now.isoformat(),deployment_host='server2/openlibing-DevEnv-055409',scope_count=109,counts=r['counts'],remaining_unresolved=r['remaining_unresolved'],factory_mappings=mappings,changed_shared_files=files,private_directory=str(W),patch='provider.patch',backup='provider_init.backup.py',runtime_recipe='frozen_v1/chunk_package/RUNTIME_RECIPE.json',runtime_source_binary_pins='frozen_v1/chunk_package/RUNTIME_PINS.json',freeze_proof='FREEZE_PROOF.json',device_free_registration_check='STATIC_REGISTRATION.json',unresolved=[x for x in r['rows'] if x['status']!='completed'],timings={'elapsed_work_seconds':(now-datetime.datetime.fromisoformat(start)).total_seconds(),'guard_elapsed_sum_seconds':sum(x['elapsed_seconds'] for x in receipts),'guard_calls':len(receipts),'queue_seconds':None,'device_seconds':None,'timing_note':'Per-case device_seconds is wall time of full guarded checker including CPU reference, not kernel latency. First-case queue_seconds includes launch/import overhead and is an upper bound. Shared batch elapsed must not be summed once per key.','numerical_repeats_of_completed_keys':0},limitations=['SSD original full output fails original tolerance; no mathematical/kernel edits.','CausalConv1d original host package lacks dynamic kernel JSON.','Delta unchanged FLA oracle SDK9.2 installation metadata missing.','FP16 original ChunkKdaFwd package remains unbuilt after prior300s codegen timeout.','Large cases retain original resource admission envelope; raw current capacity saved.','Source-changed checker receipt is not credited despite passing numerical outputs.'])
(W/'DELIVERY.json').write_text(json.dumps(d,indent=2));print(r['counts'],r['remaining_unresolved'])
