import pathlib,shutil,json,hashlib,struct,os
p=pathlib.Path('/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_bmm1');p.mkdir(parents=True,exist_ok=True)
w=pathlib.Path('/opt/tileops-all-pairs/workspace');f=p/'frozen_workspace';f.mkdir(exist_ok=True)
for name in ['benchmarks','workloads','src','scripts']:
 shutil.copytree(w/'TileOPs'/name,f/'TileOPs'/name,ignore=shutil.ignore_patterns('__pycache__','.git'),dirs_exist_ok=True)
shutil.copytree(w/'tools',f/'tools',ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
v=w/'reports/baseline-integration-700-20261026/transformer_dense/gemm/private_provider'
shutil.copytree(v,p/'private_provider',dirs_exist_ok=True)
b=p/'private_provider/op_api/lib/libcust_opapi.so';raw=b.read_bytes();assert int.from_bytes(raw[18:20],'little')==183
meta={'machine':'aarch64','elf_machine':183,'binary_size':len(raw),'binary_sha256':hashlib.sha256(raw).hexdigest(),'original_provider':str(v),'snapshot':str(p/'private_provider'),'disk_free':shutil.disk_usage(p).free}
(p/'ABI.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta))
