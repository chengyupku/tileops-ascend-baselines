import json,hashlib
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
  B,S,H,K=q.shape;HV,V=v.shape[-2:];bounds=None if offsets is None else (offsets if offsets_cpu is None else offsets_cpu).tolist();N=B if bounds is None else len(bounds)-1
  gate=g.float()[...,None].expand(B,S,HV,K).contiguous()
  state=None if initial is None else initial.contiguous().clone()
  out=torch.empty_like(v);final=torch.empty((N,HV,V,K) if p.get('state_v_first',False) else (N,HV,K,V),device=q.device,dtype=torch.float32)
  aqk=torch.empty((B,HV,S,64),device=q.device,dtype=q.dtype);akk=torch.empty_like(aqk)
  t.call([q.contiguous(),k.contiguous(),v.contiguous(),gate,beta.float().contiguous(),None,None,state,bounds,None],[out,final,None,aqk,akk,None,None,None,None,None,None],[b'BSND',float(K**-.5 if p['scale'] is None else p['scale']),64,False,-5.,False,bool(p.get('state_v_first',False))])
  return out,final
 hashes={x['path']:x['sha256'] for x in json.loads((HERE/'PINS.json').read_text())}
 for n in ['binding.py','direct_acl.py']:hashes[str(HERE/n)]=hashlib.sha256((HERE/n).read_bytes()).hexdigest()
 return Candidate(Implementation(run),'ops-transformer::aclnnChunkKdaFwd','Original official9.2 full API; lossless gate/beta FP32 binding, scalar gate broadcast, full attention and FP32 final state.',hashes,timing_mode='host_roundtrip',cleanup=t.cleanup)
