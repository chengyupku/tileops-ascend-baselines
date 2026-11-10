import os,sys,json,runpy,types,ctypes,dataclasses
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'TileOPs'),str(ROOT/'TileOPs/src'),str(ROOT)]
sys.path.insert(0,str(ROOT/'reports/baseline-cache-20261027/python'))
out=Path(sys.argv[sys.argv.index('--out')+1]);out.mkdir(parents=True,exist_ok=True)
meta=json.load(open('/run/lock/ascend-npu/device-0.lock'));assert meta['phase']=='running' and meta['child']['pid']==os.getpgrp()
vendor=Path(os.environ['OPS_NN_VENDOR'])
ctypes.CDLL('/usr/local/Ascend/cann-9.1.0/lib64/libes_nn.so',mode=ctypes.RTLD_GLOBAL)
ctypes.CDLL(str(vendor/'op_proto/es/lib/linux/aarch64/libes_nn.so'),mode=ctypes.RTLD_GLOBAL)
from benchmarks.ascend_all_pairs import registry
name='benchmarks.ascend_all_pairs.ops_nn';pkg=types.ModuleType(name);pkg.__path__=[str(ROOT/'reports/baseline-old1076-20261029/arm_nn_adapters')];pkg.__package__=name;sys.modules[name]=pkg
from benchmarks.ascend_all_pairs.ops_nn import staging,graph
orig=graph.GraphCall.__init__
def init(self,library):
 orig(self,library)
 opt=self.compiler.aclSetCompileopt;opt.argtypes=[ctypes.c_int,ctypes.c_char_p];opt.restype=ctypes.c_int
 self.check(opt(8,os.environ['BMM_CROSS_CACHE'].encode()),'isolated diagnostic CACHE_DIR')
 self.check(opt(6,str(out/'debug').encode()),'diagnostic DEBUG_DIR')
 self.check(opt(16,b'dump_cce'),'diagnostic dump_cce')
graph.GraphCall.__init__=init
original_call=graph.GraphCall.call
calls=[]
def call(self,op,inputs,outputs,attrs,**kwargs):
 import hashlib
 def describe(t):
  if t is None:return None
  host=t.detach().cpu().contiguous()
  return {'shape':list(t.shape),'stride':list(t.stride()),'dtype':str(t.dtype),'sha256':hashlib.sha256(host.reshape(-1).view(self.torch.uint8).numpy().tobytes()).hexdigest()}
 calls.append({'op':op,'inputs':[describe(t) for t in inputs],'outputs':[{'shape':list(t.shape),'dtype':str(t.dtype)} for t in outputs],'attrs':attrs,'kwargs':kwargs,'vendor':str(self.vendor)})
 (out/'CALLS.json').write_text(json.dumps(calls,indent=2))
 return original_call(self,op,inputs,outputs,attrs,**kwargs)
graph.GraphCall.call=call
(out/'ENV.json').write_text(json.dumps({k:v for k,v in os.environ.items() if k.startswith(('ASCEND','CANN','OPS_NN','TE_','TBE_','PYTHONPATH','LD_LIBRARY_PATH'))},indent=2))
from benchmarks.ascend_baselines import source_hashes
def factory(case,lib):
 def make():
  c=staging.CANDIDATES['BmmFwdOp'](case)
  return dataclasses.replace(c,source_sha256={**c.source_sha256,**source_hashes(Path(__file__))})
 return make
registry.candidate_factory=factory
out=Path(sys.argv[sys.argv.index('--out')+1]);out.mkdir(parents=True,exist_ok=True)
sys.argv=['cache_frozen_pair_check.py','--op','BmmFwdOp','--case-id','square-128-float16','--library','ops-nn','--output',str(out/'correctness.json')]
runpy.run_path(str(ROOT/'TileOPs/scripts/cache_frozen_pair_check.py'),run_name='__main__')
