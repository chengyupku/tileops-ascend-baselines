import pathlib,shutil,hashlib,json
W=pathlib.Path('/opt/tileops-all-pairs/workspace');P=W/'reports/baseline-integration-700-20261026/threehost_tla_other54_r2';records=[]
for rel in ['workloads','src/tileops/manifest']:
 for f in (W/'TileOPs'/rel).rglob('*'):
  if not f.is_file() or '__pycache__' in f.parts:continue
  d=P/'frozen'/f.relative_to(W/'TileOPs');d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,d);records.append(dict(original=str(f),private=str(d),sha256=hashlib.sha256(d.read_bytes()).hexdigest()))
(P/'FREEZE_WORKLOADS.json').write_text(json.dumps(records,indent=2));print(len(records))
