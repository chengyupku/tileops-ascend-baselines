"""Validate original staged NN candidates under the unchanged full checker."""
import dataclasses,runpy,sys,types
from pathlib import Path
ROOT=Path('/opt/tileops-baseline-cache/workspace')
HERE=Path(__file__).resolve().parent
sys.path[:0]=['/usr/local/Ascend/cann-9.1.0/python/site-packages','/usr/local/Ascend/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe']
sys.path[:0]=[str(ROOT/'TileOPs'),str(ROOT/'TileOPs/src'),str(ROOT)]
sys.path[:0]=[str(HERE/'torch213_env'),'/opt/tileops-all-pairs/workspace/.cache/all-pairs/remote_triton_runtime/overlay/site-packages']
import ctypes,os
vendor=Path(os.environ['OPS_NN_VENDOR'])
ctypes.CDLL('/usr/local/Ascend/cann-9.1.0/lib64/libes_math.so',mode=ctypes.RTLD_GLOBAL)
ctypes.CDLL('/usr/local/Ascend/cann-9.1.0/lib64/libes_nn.so',mode=ctypes.RTLD_GLOBAL)
ctypes.CDLL(str(vendor/'op_proto/es/lib/linux/aarch64/libes_nn.so'),mode=ctypes.RTLD_GLOBAL)
import torch
assert torch.__version__.startswith('2.13.'), torch.__version__
# CPU dependency backend repair; original reference/checker and NPU kernels unchanged.
torch.backends.mkldnn.enabled=False
from benchmarks.ascend_all_pairs import registry
name='benchmarks.ascend_all_pairs.ops_nn'
package=types.ModuleType(name);package.__path__=[str(HERE/'deps/reports/baseline-old1076-20261029/arm_nn_adapters')];package.__package__=name
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
  return dataclasses.replace(c,source_sha256={**c.source_sha256,**source_hashes(Path(__file__))})
 return build

# Restore the exact historical successful GEMM top-level padding interface.
if json.loads(Path(sys.argv[sys.argv.index('--task')+1]).read_text())['key'][1]=='GemmFwdOp':
 import benchmarks.ascend_nn_aligned_gemm as aligned_gemm
 original_layout=aligned_gemm._layout
 def layout_full_columns(case,alignment):
  return original_layout(case,(1024,4096,64))
 aligned_gemm._layout=layout_full_columns

registry.candidate_factory=factory
runpy.run_path(str(HERE/'bootstrap.py'),run_name='__main__')
