from pathlib import Path
import json,hashlib,shutil,fcntl
W=Path(__file__).resolve().parent;R=W.parents[2];P=R/'TileOPs/benchmarks/ascend_all_pairs/ops_transformer'
with open('/tmp/baseline700-provider-ops_transformer.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 shutil.copyfile(P/'__init__.py',W/'provider_init.final.py')
 for f in (P/'__pycache__').glob('integration700_transformer_linear_gdn.*.pyc'):f.unlink()
d=json.loads((W/'DELIVERY.json').read_text());d['original_binary_provenance']={'build':'binary_provenance/BUILD_PROOF.json','install':'binary_provenance/INSTALL_PROOF.json','all_installed_files_unchanged':'binary_provenance/CURRENT_BINARY_VERIFICATION.json','canonical_operator_sources':'binary_provenance/CANONICAL_OPERATOR_SOURCE_CHECK.json','metadata_packaging_authority':'binary_provenance/ORIGINAL_PACKAGING_METADATA_RESOLUTION.json'}
d['provider_final_snapshot']='provider_init.final.py';d['scope_finish_reason']='All109 keys now have final verified/registered success or direct current API/oracle/deployment/resource/checker evidence. Last preflight admits no remaining cases. No pending work is represented as successful.'
(W/'DELIVERY.json').write_text(json.dumps(d,indent=2));a=json.loads((W/'ARTIFACTS.json').read_text());existing={x['path'] for x in a}
for rel in ['provider_init.final.py','binary_provenance/CANONICAL_OPERATOR_SOURCE_CHECK.json','close_delivery.py']:
 f=W/rel
 if rel not in existing:a.append({'path':rel,'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
(W/'ARTIFACTS.json').write_text(json.dumps(a,indent=2));print(d['counts'],d['timings'])
