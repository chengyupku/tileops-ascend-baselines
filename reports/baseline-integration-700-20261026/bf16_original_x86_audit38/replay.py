import sys,os,ast,json,time,hashlib,platform,types
from pathlib import Path
assert sys.flags.isolated and sys.flags.no_site
assert os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']=='0'
sys.dont_write_bytecode=True
class Block:
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch_npu','acl','tilelang','workloads','benchmarks'):raise RuntimeError('forbidden import '+fullname)
sys.meta_path.insert(0,Block())
def audit(event,args):
 if event=='ctypes.dlopen' and any(s in str(args).lower() for s in ('ascend','torch_npu','libacl','libhccl')):raise RuntimeError('forbidden dlopen')
sys.addaudithook(audit)
sys.path.insert(0,'/root/dyq/workspace/.venv/lib/python3.12/site-packages')
start=time.perf_counter(); cpu=time.process_time()
import torch
torch.set_num_threads(4);torch.set_num_interop_threads(1)
P=Path(__file__).resolve().parent;W=Path('/root/dyq/workspace');G=P.parent/'gmm_capture49/attempt01/remote_evidence'
sha=lambda b:hashlib.sha256(b).hexdigest()
meta=json.loads((G/'CAPTURE.json').read_text()); cm=json.loads((G/'REFERENCE_COMPARISON.json').read_text()); hashes={}
def load(n,dtype):
 b=(G/(n+'.bin')).read_bytes();m=meta.get(n,cm['output_hashes'].get(n));assert sha(b)==m['sha256'];hashes[n]={'sha256':sha(b),'shape':m['shape'],'path':str(G/(n+'.bin'))}
 return torch.frombuffer(bytearray(b),dtype=dtype).reshape(m['shape'])
a=load('candidate_0_input0',torch.bfloat16);b=load('candidate_0_input1',torch.bfloat16)
native=load('candidate_0_output',torch.bfloat16);arm=load('reference_0_output',torch.bfloat16);fp64=load('fp64_unrounded',torch.float64);rounded=load('fp64_rounded',torch.bfloat16)
src=W/'TileOPs/workloads/gemm.py';tree=ast.parse(src.read_text())
helper=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='matmul_fp32_reduction')
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='GemmWorkload');ref=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='ref_program')
ns={'torch':torch};exec(compile(ast.Module(body=[helper,ref],type_ignores=[]),str(src),'exec'),ns)
t=time.perf_counter();c=time.process_time();out=ns['ref_program'](types.SimpleNamespace(trans_a=False,trans_b=False),a,b);run={'wall_s':time.perf_counter()-t,'cpu_s':time.process_time()-c,'calls':1}
assert all(v.device.type=='cpu' for v in (a,b,native,arm,fp64,rounded,out));assert out.dtype==torch.bfloat16
raw=out.contiguous().view(torch.uint8).numpy().tobytes();(P/'local_original.bin').write_bytes(raw)
hashes['local_original']={'sha256':sha(raw),'shape':list(out.shape),'path':str(P/'local_original.bin')}
comparisons={}
for name,x,y in [('native_vs_local_original',native,out),('local_original_vs_native',out,native),('local_original_vs_fp64_rounded',out,rounded),('local_original_vs_fp64_unrounded',out,fp64),('local_original_vs_ARM',out,arm)]:
 d=(x.double()-y.double()).abs();bad=d>(.016+.016*y.double().abs())
 try:torch.testing.assert_close(x,y,atol=.016,rtol=.016,equal_nan=True,check_dtype=x.dtype==y.dtype);ok=True
 except AssertionError:ok=False
 comparisons[name]={'values':x.numel(),'violations_float64_threshold':int(bad.sum()),'max_abs':float(d.max()),'unequal':int((x.double()!=y.double()).sum()),'torch_assert_close':ok}
libs=sorted({l.split()[-1] for l in Path('/proc/self/maps').read_text().splitlines() if '.so' in l}); forbidden=[l for l in libs if any(v in l.lower() for v in ('ascend','torch_npu','libacl','libhccl','libruntime.so'))];imports=[n for n in sys.modules if n.split('.')[0] in ('torch_npu','acl','tilelang','workloads','benchmarks')];assert not forbidden and not imports
for n,v in [('candidate_0_input0',a),('candidate_0_input1',b)]:assert sha(v.view(torch.uint8).numpy().tobytes())==hashes[n]['sha256']
r={'accepted_added':0,'diagnostic_only':True,'seed_provenance':1311006047,'shape':[1024,1024],'hashes':hashes,'comparisons':comparisons,'reference_time':run,'total_wall_s':time.perf_counter()-start,'total_cpu_s':time.process_time()-cpu,'torch':torch.__version__,'torch_config':torch.__config__.show(),'host':platform.node(),'machine':platform.machine(),'cpu_model':next(l for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')),'cpu_flags':next(l for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('flags')),'threads':torch.get_num_threads(),'interop':torch.get_num_interop_threads(),'source_sha256':sha(src.read_bytes()),'original_ast_spans':[[helper.lineno,helper.end_lineno],[ref.lineno,ref.end_lineno]],'ast_unchanged':True,'precision_override':False,'backend_selection_changed':False,'libraries':libs,'forbidden_maps':forbidden,'forbidden_imports':imports,'all_tensors_cpu':True}
(P/'CPU_CHECKS.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:r[k] for k in ('comparisons','reference_time','torch','cpu_model','forbidden_maps','forbidden_imports')},indent=2))
