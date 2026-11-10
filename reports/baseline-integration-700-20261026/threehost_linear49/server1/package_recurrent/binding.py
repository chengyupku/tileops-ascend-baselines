"""Whole original RecurrentKda for scoped ungated BF16 DeltaNet decode contracts."""
from pathlib import Path
import json,hashlib
from direct_acl import DirectACL,require_guard
P=Path(__file__).resolve().parent

def build(case):
 require_guard()
 import torch
 from benchmarks.api import Implementation
 from benchmarks.ascend_baselines import Candidate
 assert case._call.signature.name=='DeltaNetFwdOp' and case.id in ('decode-fresh-bfloat16','decode-continued-bfloat16','decode-narrow-b17-bfloat16')
 c=json.loads((P.parent/'CONTRACTS.json').read_text())[case.id];assert json.loads(json.dumps(case.params))==c['params']
 assert not case.params.get('use_qk_l2norm_in_kernel',False)
 r=json.loads((P/'RUNTIME_RECIPE.json').read_text());t=DirectACL('RecurrentKda',torch,r['opapi'],r['nnopbase'])
 def run(q,k,v,beta,initial,offsets,offsets_cpu):
  assert offsets is None and offsets_cpu is None and q.dtype==torch.bfloat16
  B,S,H,K=q.shape;HV,V=v.shape[-2:];assert S==1 and K<=128 and V<=128
  if K<128:q=torch.nn.functional.pad(q,(0,128-K));k=torch.nn.functional.pad(k,(0,128-K))
  if V<128:v=torch.nn.functional.pad(v,(0,128-V))
  state=torch.zeros((B,HV,128,128),dtype=torch.float32,device=q.device) if initial is None else torch.nn.functional.pad(initial,(0,128-V,0,128-K)).contiguous().clone()
  gate=torch.zeros((B,S,HV,128),dtype=torch.float32,device=q.device)
  out=torch.empty_like(v);final=torch.empty_like(state)
  t.call([q.contiguous(),k.contiguous(),v.contiguous(),gate,beta.contiguous(),state,None,None,None,None,None],[out,final],[b'BSND',float(K**-.5 if case.params['scale'] is None else case.params['scale']),True,False,False,False,False,False,False,-5.,False])
  return out[...,:V].contiguous(),final[...,:K,:V].contiguous()
 pins=json.loads((P/'RUNTIME_PINS.json').read_text())['files'];hashes={x['path']:x['sha256'] for x in pins}
 for f in [P/'binding.py',P/'direct_acl.py',P/'entry.py',Path(r['opapi']),Path(r['nnopbase'])]:hashes[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
 return Candidate(Implementation(run),'ops-transformer::aclnnRecurrentKda','Original complete recurrent API; exact zero ungated log gate, zero-pad and crop only, preserved original scale and FP32 state; both outputs.',hashes,timing_mode='host_roundtrip',cleanup=t.cleanup)
