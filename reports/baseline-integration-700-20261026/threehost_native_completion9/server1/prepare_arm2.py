from pathlib import Path
import json,hashlib,shutil,sys,time
T=Path(__file__).resolve().parent; W=T.parents[2]; R=T.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
mode=sys.argv[1]; vendor=R/('transformer_dense/gemm/private_provider')
files=[W/'TileOPs'/p for p in ['benchmarks/ascend_math_more.py','benchmarks/ascend_baselines.py','scripts/ascend_pair_check.py','workloads/numerics.py','benchmarks/api.py','workloads/workload_base.py','workloads/gemm.py','benchmarks/_cases/__init__.py']]
files+=list(T.glob('*.py'))
if mode=='bf16arm2': files += [T/'arm_closure'/p for p in ['bf16_moe_hadamard_native10/adapter.py','bf16_moe_hadamard_parent10/adapter.py','bf16_local_gemm18/binding.py']]
else:files += [R/'transformer_dense/integration700_transformer_dense_fp8_bmm.py']
files += [p for p in vendor.rglob('*') if p.is_file() and ('grouped_matmul' in str(p).lower() or p.name=='libcust_opapi.so') and p.suffix in ['.py','.json','.o','.so','.cpp','.h']]
assert (vendor/'op_api/lib/libcust_opapi.so') in files
manifest={str(p):sha(p) for p in files}
(T/('FROZEN_'+mode+'.json')).write_text(json.dumps({'sha256':manifest,'vendor':str(vendor),'time':time.time()},indent=2))
for p in files:
 if p.suffix=='.py':
  d=T/'snapshots'/p.relative_to(W);d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,d)
print(mode,len(files))
