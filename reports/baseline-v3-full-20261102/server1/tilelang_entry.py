import os,sys,json,types,runpy,dataclasses
from pathlib import Path
ROOT=Path('/opt/tileops-baseline-cache/workspace')
HERE=Path(__file__).resolve().parent
s=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text())
assert s.get('phase')=='running' and s.get('child',{}).get('pid')==os.getpgrp() and s.get('owner','').startswith('v3-full/server1/')
if os.environ.get('RETEST_TK36_CLEAN')!='1':
 env=dict(os.environ);env['RETEST_TK36_CLEAN']='1'
 sdk=str(HERE/'sdk920')
 env.update(ASCEND_HOME_PATH=sdk,ASCEND_TOOLKIT_HOME=sdk,ASCEND_AICPU_PATH='/usr/local/Ascend/cann-9.1.0',ASCEND_OPP_PATH='/usr/local/Ascend/cann-9.1.0/opp')
 env['PATH']=sdk+'/tools/bisheng_compiler/bin:'+sdk+'/compiler/ccec_compiler/bin:'+env.get('PATH','')
 # Keep existing verified runtime libraries; SDK9.2 supplies compiler and headers only.
 env['LD_LIBRARY_PATH']='/opt/tileops-ci/runtime/lib/python3.12/site-packages/torch/lib:/opt/tileops-ci/runtime/lib/python3.12/site-packages/torch_npu/lib:'+':'.join(p for p in env.get('LD_LIBRARY_PATH','').split(':') if '/torch/lib' not in p and '/torch_npu/lib' not in p)
 os.execve(sys.executable,[sys.executable,'-I','-S','-B',*sys.argv],env)
sys.path[:0]=[str(ROOT/'ascend_baselines/tilelang'),'/opt/tileops-ci/runtime/lib/python3.12/site-packages',str(ROOT/'TileOPs'),str(ROOT/'TileOPs/src'),str(ROOT)]
pkg=types.ModuleType('retest_tk35');pkg.__path__=[str(HERE/'deps/reports/baseline-retest32-20261030/server2/tilelang_original_v44')];sys.modules[pkg.__name__]=pkg
import importlib
OPS={'RMSNormFwdOp': ('rms_original', 'tilelang_rms_cases'), 'GemmFwdOp': ('gemm_original', 'tilelang_gemm_cases'), 'AddFwdOp': ('more_original', 'tilelang_add'), 'FP8QuantFwdOp': ('quant_original', 'tilelang_fp8_cases'), 'BmmFwdOp': ('adapters', 'bmm'), 'GroupedGemmFwdOp': ('adapters', 'grouped_gemm'), 'Conv1dFwdOp': ('adapters', 'convolution'), 'Conv2dFwdOp': ('adapters', 'convolution'), 'Conv3dFwdOp': ('adapters', 'convolution'), 'BmmFP8FwdOp': ('fp8', 'fp8_scaled'), 'GemmFP8FwdOp': ('fp8', 'fp8_scaled'), 'MoEGroupedGemmFwdOp': ('moe_activation', 'moe_grouped_dispatch')}
# Original optional IR cache is host-runtime specific; use original builder on ARM.
arm_ir=HERE/'attention_arm_ir'
arm_ir.mkdir(exist_ok=True)
importlib.import_module('retest_tk35.attention_ir').IR_ROOT=arm_ir
from benchmarks.ascend_all_pairs import registry
from benchmarks.ascend_baselines import source_hashes
def factory(case,library):
 assert library=='tilelang'
 def build():
  name=case._call.signature.name
  if name=='GQADenseFwdOp':mod,fn=('attention' if case.inputs[0].shape[1]==1 else 'attention_crops'),'dense'
  elif name=='GQAPagedFwdOp':mod,fn=('attention' if case.inputs[0].shape[0]==case.inputs[4].numel() else 'attention_crops'),'paged'
  else:mod,fn=OPS[name]
  c=getattr(importlib.import_module('retest_tk35.'+mod),fn)(case)
  return dataclasses.replace(c,source_sha256={**c.source_sha256,**source_hashes(Path(__file__))})
 return build
registry.candidate_factory=factory
runpy.run_path(str(HERE/'bootstrap.py'),run_name='__main__')
