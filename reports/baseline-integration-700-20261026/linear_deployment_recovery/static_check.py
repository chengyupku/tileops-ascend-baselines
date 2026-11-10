from pathlib import Path
import json,hashlib,subprocess,os
W=Path('/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/linear_deployment_recovery');P=W/'package';r=json.loads((P/'RUNTIME_RECIPE.json').read_text());pins=json.loads((P/'PINS.json').read_text());issues=[]
for x in pins:
 if hashlib.sha256(Path(x['path']).read_bytes()).hexdigest()!=x['sha256']:issues.append(x['path'])
source=Path(r['official_source']); config=source/'opp/built-in/op_impl/ai_core/tbe/kernel/config/ascend950/ops_transformer/chunk_kda_fwd.json'
for target in [Path(r['opapi']),Path(r['nnopbase'])]:
 q=subprocess.run(['readelf','-h',str(target)],capture_output=True,text=True);(W/(target.name+'.elf.txt')).write_text(q.stdout+q.stderr)
files=list((source/'opp/built-in/op_impl/ai_core/tbe/kernel/ascend950/ops_transformer/chunk_kda_fwd').glob('*'))
d=dict(hash_mismatches=issues,binary_count=sum(p.suffix=='.o' for p in files),metadata_count=sum(p.suffix=='.json' for p in files),config=str(config),configured_opp_target=str((Path(r['opp'])/config.relative_to(source/'opp')).resolve()),native_api=r['opapi'],npu_imports=0)
(W/'STATIC_CHECK.json').write_text(json.dumps(d,indent=2));print(json.dumps(d))
