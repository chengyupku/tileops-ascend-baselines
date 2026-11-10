import json,hashlib
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
