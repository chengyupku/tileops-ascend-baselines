from pathlib import Path
import json,shutil,hashlib,ast
R=Path('/opt/tileops-all-pairs/workspace');W=R/'reports/baseline-integration-700-20261026/linear_deployment_recovery';P=W/'package';P.mkdir(exist_ok=True)
old=R/'.cache/all-pairs/full_rollout_20261024/remote_ready/server2/group1/resume_20261025/continuation0430/tasks/recovery0730-delta-whole-api-deployment/chunk_package'
shutil.copy2(old/'direct_acl.py',P/'direct_acl.py')
source=R/'.cache/all-pairs/resume22_server2_tilekernels_adaptive_norm02/sdk_parts/cann-950-ops-transformer';view=source.parent.parent/'sdk_view_v10';sdk=R/'.cache/all-pairs/remote_ops_math_continuation/sdk920-original-layout'
r=json.loads((R/'reports/baseline-integration-700-20261026/transformer_linear/chunk_package/RUNTIME_RECIPE.json').read_text());r.update(opapi=str(source/'aarch64-linux/lib64/libopapi_transformer.so'),nnopbase=str(view/'lib64/libnnopbase.so'),official_source=str(source),opp=str(view/'opp'))
(P/'RUNTIME_RECIPE.json').write_text(json.dumps(r,indent=2))
files=[source/'aarch64-linux/include/aclnnop/level2/aclnn_chunk_kda_fwd.h',Path(r['opapi']),Path(r['nnopbase']),source/'share/info/ops_transformer/version.info']+list((source/'opp/built-in/op_impl/ai_core/tbe/kernel/ascend950/ops_transformer/chunk_kda_fwd').glob('*'))+[source/'opp/built-in/op_impl/ai_core/tbe/kernel/config/ascend950/ops_transformer/chunk_kda_fwd.json']
(P/'PINS.json').write_text(json.dumps([dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in files],indent=2))
shutil.copy2(R/'reports/baseline-integration-700-20261026/transformer_linear/ALL_CONTRACTS.json',W/'ALL_CONTRACTS.json')
s='''import json,hashlib
from pathlib import Path
from direct_acl import DirectACL,require_guard
HERE=Path(__file__).resolve().parent
def build(case):
 require_guard()
 import torch
 from benchmarks.api import Implementation
 from benchmarks.ascend_baselines import Candidate
 c=next(x for x in json.loads((HERE.parent/'ALL_CONTRACTS.json').read_text()) if x['op']==case._call.signature.name and x['case_id']==case.id)
 assert json.loads(json.dumps(case.params))==c['params']
 p=case.params
 assert case._call.signature.name=='GDNFwdOp'
 assert not any(p.get(k) for k in ('use_qk_l2norm_in_kernel','use_beta_sigmoid_in_kernel','allow_neg_eigval','use_gate_in_kernel'))
 r=json.loads((HERE/'RUNTIME_RECIPE.json').read_text());t=DirectACL('ChunkKdaFwd',torch,r['opapi'],r['nnopbase'])
 def run(q,k,v,g,beta,initial,offsets,offsets_cpu,a_log,dt_bias):
  assert a_log is None and dt_bias is None
  B,S,H,K=q.shape;HV,V=v.shape[-2:];bounds=None if offsets is None else offsets_cpu.tolist();N=B if bounds is None else len(bounds)-1
  gate=g.float()[...,None].expand(B,S,HV,K).contiguous()
  state=None if initial is None else initial.contiguous().clone()
  out=torch.empty_like(v);final=torch.empty((N,HV,V,K) if p.get('state_v_first',False) else (N,HV,K,V),device=q.device,dtype=torch.float32)
  aqk=torch.empty((B,HV,S,64),device=q.device,dtype=q.dtype);akk=torch.empty_like(aqk)
  t.call([q.contiguous(),k.contiguous(),v.contiguous(),gate,beta.float().contiguous(),None,None,state,bounds,None],[out,final,None,aqk,akk,None,None,None,None,None,None],[b'BSND',float(K**-.5 if p['scale'] is None else p['scale']),64,False,-5.,False,bool(p.get('state_v_first',False))])
  return out,final
 hashes={x['path']:x['sha256'] for x in json.loads((HERE/'PINS.json').read_text())}
 for n in ['binding.py','direct_acl.py']:hashes[str(HERE/n)]=hashlib.sha256((HERE/n).read_bytes()).hexdigest()
 return Candidate(Implementation(run),'ops-transformer::aclnnChunkKdaFwd','Original official9.2 full API; lossless gate/beta FP32 binding, scalar gate broadcast, full attention and FP32 final state.',hashes,timing_mode='host_roundtrip',cleanup=t.cleanup)
''';ast.parse(s);(P/'binding.py').write_text(s)
entry='''import sys,os,json,runpy
from pathlib import Path
from direct_acl import require_guard
P=Path(__file__).resolve().parent;W=P.parent;R=W.parents[2]
require_guard();r=json.loads((P/'RUNTIME_RECIPE.json').read_text());sys.path[:0]=[str(R/'TileOPs'),str(R/'TileOPs/src'),str(R)];sys.path.extend(r['python_paths'])
import setuptools
import benchmarks.ascend_all_pairs.registry as registry
from binding import build
original=registry.candidate_factory
registry.candidate_factory=lambda case,lib: (lambda:build(case)) if lib=='ops-transformer' and case._call.signature.name=='GDNFwdOp' else original(case,lib)
runpy.run_path(r['checker'],run_name='__main__')
''';(P/'entry.py').write_text(entry)
d=W/'decode-b1-float16';d.mkdir(exist_ok=True)
env=dict(ASCEND_HOME_PATH=str(sdk),ASCEND_TOOLKIT_HOME=str(sdk),ASCEND_OPP_PATH=r['opp'],ASCEND_CUSTOM_OPP_PATH='',GDN_CASE_OWNER='baseline700-20261026/linear_deployment_recovery',PYTHONDONTWRITEBYTECODE='1',TORCH_DEVICE_BACKEND_AUTOLOAD='0',FLA_NPU_DISABLE_PTH='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
env['LD_LIBRARY_PATH']=':'.join([str(sdk/'lib64'),str(view/'lib64'),str(source/'aarch64-linux/lib64'),'/usr/local/Ascend/driver/lib64/driver','/usr/local/Ascend/driver/lib64/common',str(R/'.cache/all-pairs/remote_triton_runtime/overlay/site-packages/torch/lib'),str(R/'.cache/all-pairs/remote_triton_runtime/overlay/site-packages/torch_npu/lib')])
for k in ['TMPDIR','XDG_CACHE_HOME','ASCEND_PROCESS_LOG_PATH','ASCEND_WORK_PATH','ASCEND_CACHE_PATH','TORCH_EXTENSIONS_DIR','TRITON_CACHE_DIR']:
 f=d/k;f.mkdir(exist_ok=True);env[k]=str(f)
script='#!/bin/bash\nset -eo pipefail\n'+f'cd "{W}"\n'+'\n'.join(f'export {k}="{v}"' for k,v in env.items())+f'\nexec /opt/tileops-ci/runtime/bin/python -S -B "{P}/entry.py" --op GDNFwdOp --case-id decode-b1-float16 --library ops-transformer --output "{d}/checker.json"\n'
(d/'child.sh').write_text(script)
print(json.dumps({'new_change':'official9.2 full package instead of missing custom compilation','files_pinned':len(files),'opapi':r['opapi'],'opp':r['opp']}))
