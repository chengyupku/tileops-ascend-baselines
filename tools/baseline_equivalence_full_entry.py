"""Identical original NN API bindings and auditable compiler source selection."""
import os,sys,json,runpy,types,ctypes,dataclasses
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'TileOPs'),str(ROOT/'TileOPs/src'),str(ROOT)]
sys.path.insert(0,str(ROOT/'reports/baseline-cache-20261027/python'))
meta=json.load(open('/run/lock/ascend-npu/device-0.lock'));assert meta['phase']=='running' and meta['child']['pid']==os.getpgrp()
out=Path(sys.argv[sys.argv.index('--out')+1]);out.mkdir(parents=True,exist_ok=True)
vendor=Path(os.environ['OPS_NN_VENDOR']);assert vendor.name=='allpairs_nn' and (vendor/'op_impl/ai_core/tbe/allpairs_nn_impl').is_dir()
ctypes.CDLL('/usr/local/Ascend/cann-9.1.0/lib64/libes_nn.so',mode=ctypes.RTLD_GLOBAL)
ctypes.CDLL(str(vendor/'op_proto/es/lib/linux/aarch64/libes_nn.so'),mode=ctypes.RTLD_GLOBAL)
from benchmarks.ascend_all_pairs import registry
name='benchmarks.ascend_all_pairs.ops_nn';pkg=types.ModuleType(name);pkg.__path__=[str(ROOT/'reports/baseline-old1076-20261029/arm_nn_adapters')];pkg.__package__=name;sys.modules[name]=pkg
from benchmarks.ascend_all_pairs.ops_nn import staging,graph,matrix,common
from benchmarks.ascend_baselines import source_hashes
base_init=graph.GraphCall.__init__
def init(self,lib):
 base_init(self,lib)
 opt=self.compiler.aclSetCompileopt;opt.argtypes=[ctypes.c_int,ctypes.c_char_p];opt.restype=ctypes.c_int
 # Fresh compiler cache for each repetition; compilation is outside measured warm runs.
 for k,v in [(8,str(out/'compile_cache')),(6,str(out/'compile_debug')),(16,'dump_cce')]:
  Path(out/'compile_cache').mkdir(exist_ok=True);Path(out/'compile_debug').mkdir(exist_ok=True)
  self.check(opt(k,v.encode()),'experiment compilation provenance')
graph.GraphCall.__init__=init

def gemm(case):
 assert not case.params['trans_a'] and not case.params['trans_b']
 binding=common.new_binding()
 def run(a,b):
  import torch
  out=torch.empty((1,a.shape[0],b.shape[1]),device=a.device,dtype=a.dtype)
  binding.call('BatchMatMulV3',[a.contiguous().unsqueeze(0),b.contiguous().unsqueeze(0),None,None],[out],{'adj_x1':False,'adj_x2':False,'enable_hf32':False})
  return out.squeeze(0)
 c=common.candidate(binding,run,'BatchMatMulV3(batch=1)','Rank2 GEMM lifted to singleton batch using original BatchMatMulV3 public API; unchanged original kernel and full checker.','matmul/batch_mat_mul_v3')
 return dataclasses.replace(c,source_sha256={**c.source_sha256,**source_hashes(Path(matrix.__file__),Path(__file__))})

def factory(case,lib):
 assert lib=='ops-nn'
 def build():
  c=gemm(case) if case._call.signature.name=='GemmFwdOp' else staging.CANDIDATES[case._call.signature.name](case)
  return dataclasses.replace(c,source_sha256={**c.source_sha256,**source_hashes(Path(__file__))})
 return build
registry.candidate_factory=factory
(out/'DEPLOYMENT.json').write_text(json.dumps({'vendor':str(vendor),'entrypoint':str(Path(__file__))},indent=2))
runpy.run_path(str(ROOT/'tools/measure_baseline_equivalence.py'),run_name='__main__')
