from pathlib import Path
import json,hashlib
W=Path('/opt/tileops-all-pairs/workspace');R=W/'reports/baseline-integration-20261026';D=R/'register_server2/repair01';P=W/'TileOPs/benchmarks/ascend_all_pairs/ops_transformer';m=json.loads((P/'validated_manifest.json').read_text());rows=[];identity=[]
from functools import lru_cache
@lru_cache(maxsize=None)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for old in json.loads((R/'register_server2/KEY_FACTORY_RECEIPTS.json').read_text()):
 row=dict(old)
 for field in ['checker','guard']:row[field]=row[field].replace('/root/dyq/workspace',str(W))
 c=json.loads(Path(row['checker']).read_text());g=json.loads(Path(row['guard']).read_text())
 assert c['status']=='verified' and g['returncode']==0
 assert [c['library'],c['op'],c['case_id']]==row['key']
 group=m['contracts'][c['case_id']]['group'];cfg=m['groups'][group]
 row['checker_sha256']=sha(Path(row['checker']));row['guard_sha256']=sha(Path(row['guard']));row['integration_validation']='CPU registry/factory mock + AST/hash only; no new numerical receipt'
 row['factory']='benchmarks.ascend_all_pairs.'+row['factory'];rows.append(row)
 for file,h in c['source_sha256'].items():
  path=Path(file) if Path(file).is_absolute() else W/file
  # Pin complete original candidate source closure, not current provider wrappers.
  assert sha(path)==h,(path,'receipt hash differs')
  cfg['pins'][str(path)]=h
 for name,path in m['sources'].items():
  if str(path) in cfg['pins']:
   found=[h for f,h in c['source_sha256'].items() if str(Path(f) if Path(f).is_absolute() else W/f)==path]
   if found:identity.append({'case_id':c['case_id'],'source':path,'receipt_sha256':found[0],'matches':found[0]==sha(Path(path))})
(P/'validated_manifest.json').write_text(json.dumps(m,indent=2)+'\n')
(D/'KEY_FACTORY_RECEIPTS.json').write_text(json.dumps(rows,indent=2)+'\n');(D/'RECEIPT_SOURCE_IDENTITY.json').write_text(json.dumps(identity,indent=2)+'\n')
print('receipt mappings',len(rows),'implementation receipt matches',len(identity),'pins',{k:len(v['pins']) for k,v in m['groups'].items()})
