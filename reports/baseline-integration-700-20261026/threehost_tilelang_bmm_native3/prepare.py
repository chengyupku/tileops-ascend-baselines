from pathlib import Path
import shutil,json,hashlib,subprocess
W=Path('/opt/tileops-all-pairs/workspace');P=W/'reports/baseline-integration-700-20261026/threehost_tilelang_bmm_native3';O=P.parent/'threehost_tla_other54_r2'
P.mkdir(exist_ok=True);records=[]
for name in ['frozen','tilelang_adapter.py']:
 s=O/name;d=P/name
 if s.is_dir():shutil.copytree(s,d,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
 else:shutil.copyfile(s,d)
 for f in ([d] if d.is_file() else d.rglob('*')):
  if f.is_file():
   original=s if d.is_file() else s/f.relative_to(d)
   h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
   assert h(f)==h(original)
   records.append(dict(original=str(original),private=str(f),sha256=h(f)))
(P/'SNAPSHOT.json').write_text(json.dumps(records,indent=2))
sdk=W/'.cache/all-pairs/remote_native_deepgemm_deepep/sdk-920-release-headers'
files=[sdk/'dependency_receipt.json',sdk/'tools/bisheng_compiler/bin/bisheng',W/'tilelang/examples/ascend/example_gemm.py',W/'tilelang/tilelang/__init__.py',*list((W/'tilelang/tilelang/lib').glob('*.so'))]
rec=[]
for f in files:
 assert f.is_file(),str(f)
 rec.append(dict(path=str(f),resolved=str(f.resolve()),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),elf_machine=int.from_bytes(f.read_bytes()[18:20],'little') if f.read_bytes()[:4]==b'\x7fELF' else None))
(P/'CPU_PREFLIGHT.json').write_text(json.dumps(rec,indent=2));print(json.dumps(rec,indent=2))
