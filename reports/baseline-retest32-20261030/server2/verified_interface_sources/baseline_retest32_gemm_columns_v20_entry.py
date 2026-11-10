import os,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1]
v=R/'reports/baseline-retest32-20261030/server2/original_nn950_historical_v17/vendors/allpairs_nn'
if os.environ.get('RETEST32_NN950_READY')!='1':
 e=dict(os.environ);e.update(RETEST32_NN950_READY='1',OPS_NN_VENDOR=str(v),ASCEND_CUSTOM_OPP_PATH=str(v),ASCEND_CLASSIC_NN_VENDOR=str(v))
 e['LD_LIBRARY_PATH']=str(v/'op_api/lib')+':'+str(v/'op_proto/es/lib/linux/aarch64')+':'+e.get('LD_LIBRARY_PATH','')
 os.execve(sys.executable,[sys.executable,*sys.argv],e)
"""Validate original staged NN candidates under the unchanged full checker."""
import dataclasses,runpy,sys,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'TileOPs'),str(ROOT/'TileOPs/src'),str(ROOT)]
import ctypes,os
vendor=Path(os.environ['OPS_NN_VENDOR'])
ctypes.CDLL('/usr/local/Ascend/cann-9.1.0/lib64/libes_math.so',mode=ctypes.RTLD_GLOBAL)
ctypes.CDLL('/usr/local/Ascend/cann-9.1.0/lib64/libes_nn.so',mode=ctypes.RTLD_GLOBAL)
ctypes.CDLL(str(vendor/'op_proto/es/lib/linux/aarch64/libes_nn.so'),mode=ctypes.RTLD_GLOBAL)
# Environment repair: ARM oneDNN BF16 CPU bmm disagrees with FP64; original checker/reference unchanged.
import torch
torch.backends.mkldnn.enabled=False
from benchmarks.ascend_all_pairs import registry
name='benchmarks.ascend_all_pairs.ops_nn'
package=types.ModuleType(name);package.__path__=[str(ROOT/'reports/baseline-old1076-20261029/arm_nn_adapters')];package.__package__=name
sys.modules[name]=package
from benchmarks.ascend_all_pairs.ops_nn import staging
from benchmarks.ascend_all_pairs.ops_nn.normalization import ada_layer_norm,variance
staging.CANDIDATES.update(AdaLayerNormFwdOp=ada_layer_norm,VarFwdOp=variance,VarMeanFwdOp=variance)
from benchmarks.ascend_baselines import source_hashes
from benchmarks.ascend_all_pairs.ops_nn import graph
import benchmarks.ascend_graph_call as legacy_graph
legacy_graph.GraphCall=graph.GraphCall
from benchmarks.ascend_all_pairs.ops_nn.normalization import normalization
staging.CANDIDATES.update(FusedAddLayerNormFwdOp=normalization,FusedAddRMSNormFwdOp=normalization)
import json
out=Path(sys.argv[sys.argv.index('--out')+1])
base_init=graph.GraphCall.__init__
def traced_init(self,lib):
 base_init(self,lib)
 opt=self.compiler.aclSetCompileopt;opt.argtypes=[ctypes.c_int,ctypes.c_char_p];opt.restype=ctypes.c_int
 for k,v in [(8,str(out/'compile_cache')),(6,str(out/'compile_debug')),(16,'dump_cce')]:
  (out/'compile_cache').mkdir(exist_ok=True);(out/'compile_debug').mkdir(exist_ok=True)
  self.check(opt(k,v.encode()),'private provenance compile option')
 (out/'DEPLOYMENT.json').write_text(json.dumps({'vendor':str(self.vendor),'owned_ops':sorted(self.owned_ops)}))
graph.GraphCall.__init__=traced_init

def factory(case,library):
 if library!='ops-nn':raise ValueError('ops-nn staging only')
 def build():
  c=staging.CANDIDATES[case._call.signature.name](case)
  return dataclasses.replace(c,adaptation=c.adaptation.replace('1024x256x64','1024x4096x64'),source_sha256={**c.source_sha256,**source_hashes(Path(__file__))})
 return build
import benchmarks.ascend_nn_aligned_gemm as aligned_gemm
original_layout=aligned_gemm._layout
def layout_full_columns(case,alignment):
 return original_layout(case,(1024,4096,64))
aligned_gemm._layout=layout_full_columns
registry.candidate_factory=factory
runpy.run_path(str(ROOT/'tools/measure_baseline_retest32.py'),run_name='__main__')
