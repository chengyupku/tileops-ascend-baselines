from pathlib import Path
import json,hashlib,datetime
P=Path(__file__).resolve().parent
result=json.loads((P/'RESULT.json').read_text());scope=json.loads((P/'SCOPE.json').read_text())
regs=[json.loads(f.read_text()) for f in P.glob('REGISTERED*.json')]
mapping=[dict(baseline='tilelang-ascend',operator=op,case_id=cid,factory=r['factory'],module_hash=r['module_hash'],pins=r['pins']) for r in regs for op,cid in r['keys']]
done={(r['baseline'],r['operator'],r['case_id']) for r in result['rows']}
delivery=dict(host='server1',workspace='/opt/tileops-all-pairs/workspace',owner='baseline700-20261026/tla_simple_tilelang',scope_count=80,rows_count=len(result['rows']),complete_scope=len(done)==80,
 counts={s:sum(r['status']==s for r in result['rows']) for s in set(r['status'] for r in result['rows'])},
 formal_mapping=mapping,changed_shared=['TileOPs/benchmarks/ascend_all_pairs/tla/__init__.py']+['TileOPs/benchmarks/ascend_all_pairs/tla/'+f.name for f in P.glob('integration700_*.py')],
 patches=[f.name for f in P.glob('registration_*.patch')],backups=[f.name for f in P.glob('provider_init_*.bak')],
 run_recipe={'private':'python3 launch_batch.py BATCH.json (launches the full foreground command under required npu-guard, 3600s wait / 300s timeout)','formal':'Fresh /opt/tileops-all-pairs/bin/allpairs-python TileOPs/scripts/ascend_pair_check.py --library tilelang-ascend --op OP --case-id CASE --output RECEIPT, entirely inside npu-guard; no repeat needed for already passed keys','registry_check':'python3 check_registry.py (device-free)'},
 remaining_unattempted=[{k:r[k] for k in ['baseline','operator','case_id']} for r in scope if (r['baseline'],r['operator'],r['case_id']) not in done],
 unresolved=[r for r in result['rows'] if r['status'] not in ['completed','already_completed']],
 no_benchmark=True,no_vendor_kernel_or_compiler_edits=True,
 timing_note='Remote and local clocks differ by approximately 93 seconds; durations are calculated within one host, queue receipt metadata retained. Batch-held time and per-checker time are not additive.',
 utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(P/'DELIVERY.json').write_text(json.dumps(delivery,indent=2))
