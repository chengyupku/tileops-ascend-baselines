from pathlib import Path
import json,shutil,ast,hashlib
W=Path(__file__).resolve().parent;R=W.parents[2];old=R/'.cache/all-pairs/resume22_server2_transformer_dense_state03';native=old/'native_build/causal';vendor=native/'extracted/packages/vendors/causalpairs_transformer';P=W/'causal_package';P.mkdir(exist_ok=True)
s=(W/'chunk_package/direct_acl.py').read_text().replace("('RecurrentKda', 'ChunkGatedDeltaRule')","('RecurrentKda', 'ChunkGatedDeltaRule', 'CausalConv1dUpdate')").replace("if op == 'RecurrentKda':", "if op == 'CausalConv1dUpdate':\n            signature = [P] * 9 + [C.c_char_p, C.c_int64, C.c_int64, P, U64P, PP]\n        elif op == 'RecurrentKda':")
(P/'direct_acl.py').write_text(s);ast.parse(s)
r=json.loads((W/'chunk_package/RUNTIME_RECIPE.json').read_text());r['opapi']=str(vendor/'op_api/lib/libcust_opapi.so');r['nnopbase']='/usr/local/Ascend/cann-9.1.0/lib64/libnnopbase.so';r['vendor']=str(vendor);r['required_environment'].update(ASCEND_HOME_PATH='/usr/local/Ascend/cann-9.1.0',ASCEND_TOOLKIT_HOME='/usr/local/Ascend/cann-9.1.0',ASCEND_OPP_PATH='/usr/local/Ascend/cann-9.1.0/opp',ASCEND_CUSTOM_OPP_PATH=str(vendor))
r['python_paths']=[x for x in r['python_paths'] if 'sdk920' not in x and '/gdn-original-install' not in x]+['/usr/local/Ascend/cann-9.1.0/python/site-packages','/usr/local/Ascend/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe'];(P/'RUNTIME_RECIPE.json').write_text(json.dumps(r,indent=2))
s='''import json,hashlib
from pathlib import Path
from direct_acl import DirectACL,require_guard
HERE=Path(__file__).resolve().parent
def build(case):
 require_guard()
 import torch
 from benchmarks.api import Implementation
 from benchmarks.ascend_baselines import Candidate
 r=json.loads((HERE/'RUNTIME_RECIPE.json').read_text())
 contracts=json.loads((HERE.parent/'ALL_CONTRACTS.json').read_text());c=next(x for x in contracts if x['op']=='CausalConv1dUpdateFwdOp' and x['case_id']==case.id)
 assert json.loads(json.dumps(case.params))==c['params']
 transport=DirectACL('CausalConv1dUpdate',torch,r['opapi'],r['nnopbase'])
 def run(x,weight,conv_state,bias):
  packed=x[:,None,:].contiguous();state=conv_state[...,-(weight.shape[-1]-1):].transpose(1,2).contiguous().clone();out=torch.empty_like(packed)
  transport.call([packed,weight.T.contiguous(),state,None if bias is None else bias.contiguous(),None,None,None,None,None],[out],[(case.params['activation'] or 'none').encode(),-1,1])
  return out[:,0,:],torch.cat((conv_state[...,1:2],state.transpose(1,2)),dim=-1).contiguous()
 hashes={x['path']:x['sha256'] for x in json.loads((HERE/'PINS.json').read_text())}
 for n in ['binding.py','direct_acl.py']:hashes[str(HERE/n)]=hashlib.sha256((HERE/n).read_bytes()).hexdigest()
 return Candidate(Implementation(run),'ops-transformer::aclnnCausalConv1dUpdate','Original whole update API; transpose W/D and history, preserve complete returned state and original input window.',hashes,timing_mode='host_roundtrip',cleanup=transport.cleanup)
''';(P/'binding.py').write_text(s);ast.parse(s)
files=[Path(r['opapi']),Path(r['nnopbase']),native/'NATIVE_DEPLOYMENT.json']+list((native/'source/mamba/causal_conv1d/op_api').glob('*'))
(P/'PINS.json').write_text(json.dumps([dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in files if f.is_file()],indent=2))
entry='''import sys,os,json,runpy
from pathlib import Path
from direct_acl import require_guard
P=Path(__file__).resolve().parent;W=P.parent;R=W.parents[2]
require_guard();r=json.loads((P/'RUNTIME_RECIPE.json').read_text());sys.path[:0]=[str(R/'TileOPs'),str(R/'TileOPs/src'),str(R)];sys.path.extend(r['python_paths'])
import setuptools
import benchmarks.ascend_all_pairs.registry as registry
from binding import build
original=registry.candidate_factory
registry.candidate_factory=lambda case,lib: (lambda:build(case)) if lib=='ops-transformer' and case._call.signature.name=='CausalConv1dUpdateFwdOp' else original(case,lib)
runpy.run_path(r['checker'],run_name='__main__')
''';(P/'entry.py').write_text(entry)
template=(W/'q35-397b-tp8-2048-bfloat16/child.sh').read_text()
for i in ['mamba2-2.7b-decode-b1-bfloat16','qwen3-next-decode-b64-float16']:
 d=W/i;d.mkdir(exist_ok=True);s=template.replace('q35-397b-tp8-2048-bfloat16',i).replace('chunk_package/entry.py','causal_package/entry.py').replace('--op GDNFwdOp','--op CausalConv1dUpdateFwdOp')
 lines=s.splitlines();pos=next(j for j,l in enumerate(lines) if l.startswith('exec /opt'))
 env=[f'export {k}="{v}"' for k,v in r['required_environment'].items()]+['export LD_LIBRARY_PATH="/usr/local/Ascend/cann-9.1.0/lib64:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:'+str(vendor/'op_api/lib')+':'+str(R/'.cache/all-pairs/remote_triton_runtime/overlay/site-packages/torch/lib')+':'+str(R/'.cache/all-pairs/remote_triton_runtime/overlay/site-packages/torch_npu/lib')+'"']
 lines[pos:pos]=env;s='\n'.join(lines)+'\n'
 for line in s.splitlines():
  if line.startswith('export ') and '="'+str(d)+'/' in line:Path(line.split('="')[1].rstrip('"')).mkdir(exist_ok=True)
 (d/'child.sh').write_text(s)
print('causal ready')
