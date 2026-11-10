from pathlib import Path
import json,hashlib,tarfile,time,difflib
P=Path('/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_attention_moe53')
rows=[]
for O in sorted(P.glob('mla-*')):
 if not O.is_dir():continue
 row={'attempt':O.name}
 for n in ['checker.json','guard.json','receipt.json','native.json']:
  p=O/n
  if p.exists():row[n]=json.loads(p.read_text())
 rows.append(row)
(P/'ALL_ATTEMPTS.json').write_text(json.dumps(rows,indent=2))
manifest=json.load(open(P/'frozen/TileOPs/benchmarks/ascend_all_pairs/ops_transformer/validated_manifest.json'))
(P/'ORIGINAL_VENDOR_PINS.json').write_text(json.dumps(manifest['groups']['attention']['pins'],indent=2))
(P/'DSA_BINDING_DIFF.txt').write_text(''.join(difflib.unified_diff((P/'dsa.original.py').read_text().splitlines(True),(P/'dsa.py').read_text().splitlines(True),fromfile='original_dsa.py',tofile='private_dsa.py')))
allpins={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in P.rglob('*') if p.is_file() and p.suffix not in ['.gz'] and '__pycache__' not in str(p)}
(P/'REMOTE_ARTIFACT_PINS.json').write_text(json.dumps(allpins,indent=2))
with tarfile.open(P/'evidence.tar.gz','w:gz') as t:
 for p in sorted(P.iterdir()):
  if p.name not in ['evidence.tar.gz']:t.add(p,arcname=p.name,recursive=True)
print('archive', (P/'evidence.tar.gz').stat().st_size,'files',len(allpins))
