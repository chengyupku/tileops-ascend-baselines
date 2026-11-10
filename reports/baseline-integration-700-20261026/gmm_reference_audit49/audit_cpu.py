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
y=w.ref_program(a,b); z64=a.double()@b.double();z=z64.bfloat16(); f=(a.float()@b.float()).bfloat16()
def compare(x,target,label):
 delta=(x.double()-target.double()).abs();limit=.016+.016*target.double().abs();bad=delta>limit;ids=bad.flatten().nonzero().flatten()
 data={'flat_index':ids.numpy(),'actual':x.flatten()[ids].float().numpy(),'target':target.flatten()[ids].float().numpy(),'abs_diff':delta.flatten()[ids].numpy(),'limit':limit.flatten()[ids].numpy()}
 np.savez_compressed(P/(label+'.npz'),**data)
 return {'elements':x.numel(),'violations':int(bad.sum()),'different':int((x!=target).sum()),'max_abs':float(delta.max()),'max_abs_index':int(delta.flatten().argmax()),'per_violation_artifact':label+'.npz','first5':[{k:v[i].item() for k,v in data.items()} for i in range(min(5,len(ids)))]}
out={'cpu_only':True,'historical_same_input':False,'canonical_acceptance':False,'torch_version':torch.__version__,'torch_git_version':torch.version.git_version,'torch_file':torch.__file__,'architecture':platform.machine(),'torch_config':torch.__config__.show(),'threads':torch.get_num_threads(),'tolerance':{'atol':.016,'rtol':.016,'rule':'abs(actual-target) <= atol+rtol*abs(target); historical checker target is original reference'},'reference_formula':'BF16 CPU torch.matmul(a,b), no FP32 conversion; CUDA reduction flags temporarily False do not specify CPU accumulation','fp64_formula':'sum_k float64(a[i,k])*float64(b[k,j]), then BF16 output rounding','reference_vs_fp64rounded':compare(y,z,'reference_vs_fp64rounded'),'fp64rounded_vs_reference_checker_direction':compare(z,y,'fp64rounded_vs_reference'),'fp32rounded_vs_fp64rounded':compare(f,z,'fp32rounded_vs_fp64rounded'),'fp32rounded_vs_reference_checker_direction':compare(f,y,'fp32rounded_vs_reference'),'output_sha256':{k:th(v) for k,v in [('original_reference',y),('fp64_unrounded',z64),('fp64_rounded',z),('fp32_rounded',f)]},'torch_npu_imported':'torch_npu' in sys.modules,'loaded_ascend_libraries':[l for l in Path('/proc/self/maps').read_text().splitlines() if any(t in l.lower() for t in ['libascend','libacl','libhccl','libtorch_npu'])]}
assert not out['torch_npu_imported'] and not out['loaded_ascend_libraries']
(P/'CANONICAL_INPUT_PROOF.json').write_text(json.dumps(proof,indent=2)+'\n');(P/'REFERENCE_COMPARISON.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'proof':proof,'comparison':out},indent=2))
