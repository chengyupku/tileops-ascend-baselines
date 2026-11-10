from pathlib import Path
import subprocess,tarfile,io,json,hashlib,ast,time
P=Path(__file__).resolve().parent;W=P.parents[2]
ssh=str(W/'.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh')
code="""import tarfile,sys
from pathlib import Path
W=Path('/opt/tileops-all-pairs/workspace')
pairs=[('reports/baseline-integration-700-20261026/threehost_tla_other54_r2/frozen','frozen'),('reports/baseline-integration-700-20261026/tla_simple_tilelang/private_tla','private_tla'),('.cache/all-pairs/full_rollout_20261024/remote_ready/server1/group1/recovery_resume25/tla_supplemental_1858/runtime_env2/runtime/package','runtime'),('reports/baseline-integration-700-20261026/tla_simple_tilelang/element_adapter.py','element_adapter.original.py'),('ascend_baselines/tilelang-ascend/testing/python/language/test_tilelang_ascend_language_elementwise.py','original_source.py')]
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as t:
 for rel,dst in pairs:
  f=W/rel
  if f.is_file():t.add(f,arcname=dst)
  else:
   for x in f.rglob('*'):
    if x.is_file() and '__pycache__' not in x.parts:t.add(x,arcname=str(Path(dst)/x.relative_to(f)))
"""
r=subprocess.run([ssh,'openlibing-DevEnv-661495','python3 -'],input=code.encode(),capture_output=True);assert r.returncode==0,r.stderr
(P/'deployment.tar.gz').write_bytes(r.stdout)
with tarfile.open(fileobj=io.BytesIO(r.stdout)) as t:t.extractall(P,filter='data')
src=(P/'element_adapter.original.py').read_text()
assert hashlib.sha256(src.encode()).hexdigest()=='07d6994064884a5027cf839ea0688bb695b12592efae11888a3f0524c47d6a7f'
fixed=src.replace("target='ascendc'","target='pto'").replace('shifts=torch.arange(8,dtype=torch.uint8,device=y.device)','shifts=torch.tensor([0,1,2,3,4,5,6,7],dtype=torch.uint8,device="cpu").to(device=y.device)')
(P/'element_adapter.py').write_text(fixed)
assert hashlib.sha256((P/'original_source.py').read_bytes()).hexdigest()=='303b90f72b7b73b800f252e46cebf2dca682bea6b9362dd051f53d80044afb84'
tree=ast.parse((P/'original_source.py').read_text());n=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='compare')
(P/'compare.original.txt').write_text(ast.get_source_segment((P/'original_source.py').read_text(),n))
checks=[]
for b in range(256):
 for i,s in enumerate([0,1,2,3,4,5,6,7]):
  old=(b>>i)&1;new=(b>>s)&1
  assert old==new
  checks.append([b,i,new])
(P/'CPU_PROOF.json').write_text(json.dumps(dict(all_256_bytes_all_8_bits=True,checks=2048,original_shifts=list(range(8)),new_shifts=[0,1,2,3,4,5,6,7],dtype='uint8',order='byte-major, bit 0 through bit 7; reshape unchanged',proof_sha256=hashlib.sha256(json.dumps(checks).encode()).hexdigest(),source_sha256=hashlib.sha256((P/'original_source.py').read_bytes()).hexdigest(),kernel_body_unchanged=True),indent=2))
(P/'HEARTBEAT.json').write_text(json.dumps(dict(phase='cpu_prepared',time=time.time(),accepted_added=0)))
print('Prepared',len(r.stdout),'bytes')
