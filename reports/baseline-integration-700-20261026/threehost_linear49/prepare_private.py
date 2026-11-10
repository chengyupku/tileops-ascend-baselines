from pathlib import Path
import shutil,json,hashlib,datetime,tarfile
P=Path(__file__).resolve().parent; W=P.parents[2];S=P/'snapshot'
for n in ['benchmarks','workloads','src','scripts']:
 shutil.copytree(W/'TileOPs'/n,S/'TileOPs'/n,ignore=shutil.ignore_patterns('__pycache__','*.pyc','.git'),dirs_exist_ok=True)
(S/'tools').mkdir(exist_ok=True);shutil.copy2(W/'tools/baseline_protocol.py',S/'tools/baseline_protocol.py')
B=P/'package';B.mkdir(exist_ok=True)
for n in ['binding.py','direct_acl.py']:
 shutil.copy2(P.parent/'transformer_linear/chunk_package'/n,B/n)
c=json.loads((P.parent/'transformer_linear/ALL_CONTRACTS.json').read_text());c={x['case_id']:x for x in c if x['op']=='DeltaNetFwdOp'};
scope=json.loads((P/'SCOPE.json').read_text());keys=[x['case_id'] for x in scope if x['baseline']=='ops-transformer' and x['operator']=='DeltaNetFwdOp'];(P/'CHUNK_CASES.json').write_text(json.dumps(keys));(P/'CONTRACTS.json').write_text(json.dumps({k:c[k] for k in keys},indent=2))
r={'opapi':'/usr/local/Ascend/cann-9.1.0/lib64/libopapi.so','nnopbase':'/usr/local/Ascend/cann-9.1.0/lib64/libnnopbase.so'};(B/'RUNTIME_RECIPE.json').write_text(json.dumps(r,indent=2));(B/'RUNTIME_PINS.json').write_text('{"files": []}')
(P/'SNAPSHOT_HASHES.json').write_text(json.dumps({str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in S.rglob('*') if p.is_file()},indent=2))
(P/'HEARTBEAT.json').write_text(json.dumps(dict(phase='private_snapshot_and_sdk91_binding',currentkey='ops-transformer/DeltaNetFwdOp/decode-fresh-bfloat16',host='server1_candidate_due_server2_busy_sample',lastprogress=datetime.datetime.now(datetime.timezone.utc).isoformat(),blockedreason=None),indent=2))
