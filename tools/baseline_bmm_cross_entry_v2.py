import os,sys,json,runpy,types,ctypes,dataclasses
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'TileOPs'),str(ROOT/'TileOPs/src'),str(ROOT)]
sys.path.insert(0,str(ROOT/'reports/baseline-cache-20261027/python'))
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
graph.GraphCall.__init__=init
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
