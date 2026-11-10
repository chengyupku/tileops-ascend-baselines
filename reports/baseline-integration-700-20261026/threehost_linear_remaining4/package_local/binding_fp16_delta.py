"""Exact scoped DeltaNet FP16 through the original complete ChunkKdaFwd API."""
from pathlib import Path
import json,hashlib
from direct_acl_kda import DirectACL,require_guard
P=Path(__file__).resolve().parent

def build(case):
 require_guard()
 import torch
 from benchmarks.api import Implementation
 from benchmarks.ascend_baselines import Candidate
 assert case._call.signature.name=='DeltaNetFwdOp' and case.id=='decode-continued-float16'
 c=json.loads((P.parent/'CONTRACTS.json').read_text())[case.id];assert json.loads(json.dumps(case.params))==c['params']
 assert not case.params.get('use_qk_l2norm_in_kernel',False)
 opapi='/usr/local/Ascend/cann-9.2.0-beta.2/lib64/libopapi_transformer.so';base='/usr/local/Ascend/cann-9.2.0-beta.2/lib64/libnnopbase.so'
 t=DirectACL('ChunkKdaFwd',torch,opapi,base)
 def run(q,k,v,beta,initial,offsets,offsets_cpu):
  assert offsets is None and offsets_cpu is None and q.dtype==torch.float16
  B,S,H,K=q.shape;HV,V=v.shape[-2:]
  # Ungated DeltaNet is the original full KDA contract at exact constant log-gate zero.
  gate=torch.zeros((B,S,HV,K),device=q.device,dtype=torch.float32)
  state=None if initial is None else initial.contiguous().clone()
  out=torch.empty_like(v);final=torch.empty((B,HV,K,V),device=q.device,dtype=torch.float32)
  aqk=torch.empty((B,HV,S,64),device=q.device,dtype=q.dtype);akk=torch.empty_like(aqk)
  t.call([q.contiguous(),k.contiguous(),v.contiguous(),gate,beta.float().contiguous(),None,None,state,None,None],[out,final,None,aqk,akk,None,None,None,None,None,None],[b'BSND',float(K**-.5 if case.params['scale'] is None else case.params['scale']),64,False,-5.,False,False])
  return out,final
 hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(opapi),Path(base),P/'binding_fp16_delta.py',P/'direct_acl_kda.py']}
 return Candidate(Implementation(run),'ops-transformer::aclnnChunkKdaFwd','Original complete FP16 API, exact ungated constant-zero log gate, lossless beta widening and full FP32 state; no BF16-only API used.',hashes,timing_mode='host_roundtrip',cleanup=t.cleanup)
