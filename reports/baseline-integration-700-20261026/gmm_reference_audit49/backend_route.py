"""CPU diagnostic only. No canonical historical-input or integration claim."""
import os, sys, ast, json, hashlib, platform
from pathlib import Path
from types import SimpleNamespace
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
class BlockNpu:
 def find_spec(self, fullname, path=None, target=None):
  if fullname == 'torch_npu' or fullname.startswith('torch_npu.'):
   raise RuntimeError('NO_NPU')
sys.meta_path.insert(0,BlockNpu())
sys.path[:0]=['/opt/tileops-all-pairs/workspace/.cache/all-pairs/remote_triton_runtime/overlay/site-packages','/opt/tileops-ci/runtime/lib/python3.12/site-packages']
import torch
import numpy as np
assert torch.version.cuda is None
W=Path('/opt/tileops-all-pairs/workspace'); P=W/'reports/baseline-integration-700-20261026/gmm_reference_audit49';P.mkdir(exist_ok=True)
D=W/'reports/baseline-integration-700-20261026/transformer_dense'
receipts={k:json.loads((D/v/'representative.json').read_text()) for k,v in [('bf16','gmm_check_ownbins'),('fp32','gmm_check_fp32')]}
r=receipts['bf16']; snap=r['case_snapshot'];params=snap['parameters']
assert snap==receipts['fp32']['case_snapshot']
def sha(b):return hashlib.sha256(b).hexdigest()
def extract(path,names,ns):
 tree=ast.parse(path.read_text()); nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]; assert len(nodes)==len(names)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
 return {n.name:sha(ast.get_source_segment(path.read_text(),n).encode()) for n in nodes}
ns={'torch':torch,'Any':object,'WorkloadBase':object,'run_device':lambda:'cpu'}
source=W/'TileOPs/workloads/gemm.py';function_hashes=extract(source,{'GemmWorkload','matmul_fp32_reduction'},ns)
seedns={'hashlib':hashlib};extract(W/'tools/baseline_protocol.py',{'seed_case'},seedns)
seed=seedns['seed_case'](torch,r['op'],r['case_id']);assert seed==r['input_seed']==receipts['fp32']['input_seed']
assert sha(source.read_bytes())==r['oracle_source_sha256']['TileOPs/workloads/gemm.py']
call=SimpleNamespace(indices={'M':params['a'][0][0],'N':params['b'][0][1],'K':params['a'][0][1],'T':params['a'][1],'trans_a':params['trans_a'],'trans_b':params['trans_b']})
w=ns['GemmWorkload'].from_call(call)
torch.set_num_threads(4)
a,b=w.gen_inputs()
def th(x):return sha(x.contiguous().view(torch.uint8).numpy().tobytes())
proof={'seed':seed,'case_snapshot':snap,'scope_count':49,'original_generator':'AST-executed unchanged GemmWorkload.from_call and gen_inputs, run_device explicitly CPU','function_sha256':function_hashes,'historical_generation_device':'npu (checker set_run_device and gen_inputs randn device argument)','historical_input_identity_established':False,'historical_input_hashes':None,'reason':'Historical receipts contain only metadata; same seed on CPU does not reproduce NPU RNG stream. No retained tensor artifacts in two representative run trees.','cpu_inputs':{k:{'sha256_raw_contiguous_bytes':th(v),'shape':list(v.shape),'dtype':str(v.dtype),'device':str(v.device),'bytes':v.numel()*v.element_size()} for k,v in [('a',a),('b',b)]},'source_sha256':{str(p.relative_to(W)):sha(p.read_bytes()) for p in [source,W/'TileOPs/workloads/device.py',W/'TileOPs/benchmarks/_cases/gemm.py',W/'TileOPs/benchmarks/api.py',W/'tools/baseline_protocol.py']},'receipt_sha256':{k:sha((D/v/'representative.json').read_bytes()) for k,v in [('bf16','gmm_check_ownbins'),('fp32','gmm_check_fp32')]}}

y=w.ref_program(a,b)
print(json.dumps({'purpose':'trace unchanged CPU reference backend, no acceptance check','output_sha256':th(y),'mkldnn_enabled':torch.backends.mkldnn.enabled,'mkldnn_available':torch.backends.mkldnn.is_available(),'loaded_blas':[l for l in Path('/proc/self/maps').read_text().splitlines() if 'blas' in l.lower()],'no_npu_import':'torch_npu' not in sys.modules}))
