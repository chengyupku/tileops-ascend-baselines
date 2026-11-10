"""CPU-only diagnostic, not canonical acceptance; blocks torch_npu imports."""
import sys, os, json, time, ast, hashlib
from pathlib import Path
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
class NoNpu:
 def find_spec(self, fullname, path=None, target=None):
  if fullname == 'torch_npu' or fullname.startswith('torch_npu.'):
   raise RuntimeError('CPU diagnostic forbids torch_npu import')
sys.meta_path.insert(0,NoNpu())
sys.path[:0]=['/opt/tileops-all-pairs/workspace/.cache/all-pairs/remote_triton_runtime/overlay/site-packages','/opt/tileops-ci/runtime/lib/python3.12/site-packages']
import torch
start=time.monotonic();torch.set_num_threads(4);torch.manual_seed(1311006047)
a=torch.randn(1024,1024,dtype=torch.bfloat16);b=torch.randn(1024,1024,dtype=torch.bfloat16)
source=Path('/opt/tileops-all-pairs/workspace/TileOPs/workloads/gemm.py'); tree=ast.parse(source.read_text()); fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='matmul_fp32_reduction'); ns={'torch':torch};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),ns); y=ns['matmul_fp32_reduction'](a,b);z=(a.double()@b.double()).bfloat16();f=(a.float()@b.float()).bfloat16()
def compare(x,z):
 delta=(x.float()-z.float()).abs(); bad=delta>0.016+0.016*z.float().abs()
 return dict(mismatches=int(bad.sum()),elements=x.numel(),max_abs=float(delta.max()),different=int((x!=z).sum()))
out=dict(cpu_only=True,canonical_acceptance=False,reason='CPU RNG differs from NPU canonical input; isolates reference arithmetic only',reference_function='unchanged workload matmul_fp32_reduction extracted via AST',reference_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),torch_version=torch.__version__,torch_file=torch.__file__,bf16_vs_fp64=compare(y,z),fp32_vs_fp64=compare(f,z),torch_npu_imported='torch_npu' in sys.modules,elapsed_s=time.monotonic()-start)
print(json.dumps(out,indent=2));Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2))
