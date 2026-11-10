from pathlib import Path
import json,hashlib,shutil
W=Path('/opt/tileops-all-pairs/workspace');R=W/'reports/baseline-integration-20261026';D=R/'register_server2/repair01';P=W/'TileOPs/benchmarks/ascend_all_pairs/ops_transformer'
D.mkdir(parents=True,exist_ok=True)
h={}
for root,label in [(P,'provider'),(R/'register_server2','first_report')]:
 for f in root.iterdir():
  if f.is_file():
   dest=D/'backup'/label/f.name;dest.parent.mkdir(parents=True,exist_ok=True)
   if not dest.exists():shutil.copy2(f,dest)
   h[str(f)]=hashlib.sha256(dest.read_bytes()).hexdigest()
(D/'BACKUP_HASHES.json').write_text(json.dumps(h,indent=2))
for f in [R/'transformer_attention/SOURCE_PINS.json',R/'transformer_attention/VENDOR.json',R/'transformer_gdn/chunk_package/RUNTIME_PINS.json']:
 d=json.loads(f.read_text());print(f.name, [(k,len(v) if isinstance(v,(list,dict)) else str(v)[:250]) for k,v in d.items()])
print((Path('/opt/tileops-all-pairs/control/private_python.py')).read_text())
