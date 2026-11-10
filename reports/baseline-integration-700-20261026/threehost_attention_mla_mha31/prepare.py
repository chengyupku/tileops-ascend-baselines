from pathlib import Path
import shutil,json,hashlib,ast
P=Path(__file__).resolve().parent;W=P.parents[2];old=P.parent/'threehost_tla_attention73'
for n in ['benchmarks','workloads','src','scripts']:
 shutil.copytree(W/'TileOPs'/n,P/'frozen/TileOPs'/n,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
shutil.copytree(old/'private_tla',P/'private_tla',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
s=(old/'run_case.py').read_text().replace('baseline700-20261026/threehost_tla_attention73','baseline700-20261026/threehost_attention_mla_mha31').replace("'-original-default-attempt02'","'-pto-A5-original-branch'").replace("os.environ.pop('TL_PLATFORM',None)","os.environ['TL_PLATFORM']='A5'")
s=s.replace('from adapter import factory','from adapter import factory\nfrom capture import install, snapshot\ninstall(out, tilelang, torch)\nsnapshot(out, "before_checker")')
(P/'run_case.py').write_text(s)
s=(old/'adapter.py').read_text().replace('target=impl.target','target="pto"').replace('platform=impl.platform','platform="A5"')
s=s.replace("result=run(*args);", "torch.save(tuple(x.detach().cpu() if isinstance(x,torch.Tensor) else x for x in args),out/'inputs.pt');result=run(*args);")
(P/'adapter.py').write_text(s)
rows={str(f.relative_to(P)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (P/'frozen').rglob('*') if f.is_file()}
(P/'FROZEN_HASHES.json').write_text(json.dumps(rows,indent=2))
for f in P.glob('*.py'):ast.parse(f.read_text())
