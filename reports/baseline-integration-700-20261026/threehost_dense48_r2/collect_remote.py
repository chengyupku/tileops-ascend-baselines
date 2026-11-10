import pathlib,json,hashlib,re
P=pathlib.Path(__file__).resolve().parent
logs=[];hits=[]
for f in (P/'run/ASCEND_PROCESS_LOG_PATH').rglob('*'):
 if f.is_file():
  logs.append({'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
  for line in f.read_text(errors='replace').splitlines():
   if ('GroupedMatmul' in line or 'grouped_matmul' in line) and any(t in line.lower() for t in ['bin','kernel','launch','tiling','register']):hits.append({'file':str(f),'line':line})
(P/'DEVICE_DISPATCH.json').write_text(json.dumps({'logs':logs,'hits':hits},indent=2))
paths=set()
for call in json.loads((P/'native.json').read_text()):
 for line in call.get('maps_after','').splitlines():
  s=line.split()
  if s and s[-1].startswith('/') and ('.so' in s[-1] or str(P/'vendor') in s[-1]):paths.add(s[-1])
result=[]
for s in sorted(paths):
 p=pathlib.Path(s)
 if p.is_file():result.append({'path':s,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'inode':p.stat().st_ino,'device':p.stat().st_dev,'size':p.stat().st_size})
(P/'LOADED_DEPENDENCIES.json').write_text(json.dumps(result,indent=2))
files=['scripts/ascend_pair_check.py','workloads/gemm.py','workloads/numerics.py','benchmarks/api.py','benchmarks/ascend_math_more.py','benchmarks/ascend_baselines.py','benchmarks/ascend_all_pairs/ops_transformer/__init__.py']
(P/'FROZEN_CHECKER.json').write_text(json.dumps([{'path':str(P/'TileOPs'/f),'sha256':hashlib.sha256((P/'TileOPs'/f).read_bytes()).hexdigest()} for f in files],indent=2))
print('dispatch lines',len(hits),'dependencies',len(result));print('\n'.join(h['line'] for h in hits if '.o' in h['line'] or '.bin' in h['line'])[-6000:])
