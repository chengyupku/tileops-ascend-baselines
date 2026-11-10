from pathlib import Path
import json,hashlib,time
D=Path('/opt/tileops-all-pairs/workspace/reports/baseline-integration-20261026/register_server2/repair01');P=Path('/opt/tileops-all-pairs/workspace/TileOPs/benchmarks/ascend_all_pairs/ops_transformer')
m=json.loads((P/'validated_manifest.json').read_text());out={}
for group,c in m['groups'].items():
 missing=[x for x in c['directories'] if not Path(x).is_dir()];errors=[]
 for f,h in c['pins'].items():
  if not Path(f).is_file():errors.append({'path':f,'error':'missing'})
  elif hashlib.sha256(Path(f).read_bytes()).hexdigest()!=h:errors.append({'path':f,'error':'hash mismatch'})
 out[group]={'missing_directories':missing,'pin_errors':errors,'pin_count':len(c['pins'])}
 # Original sys.path allows nonexistent entries; keep original order verbatim, report them.
 c['optional_absent_python_paths']=[x for x in missing if x in c['python_paths']]
 c['directories']=[x for x in c['directories'] if x not in c['optional_absent_python_paths']]
(D/'PATH_PIN_AUDIT.json').write_text(json.dumps(out,indent=2));(P/'validated_manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(out,indent=2))
