"""Whole original Update API; full length-4 state storage, layout/copy adaptation only."""
import json,hashlib,os,time
from pathlib import Path
from direct_acl import DirectACL,require_guard
HERE=Path(__file__).resolve().parent

def build(case):
 require_guard()
 import torch
 from benchmarks.api import Implementation
 from benchmarks.ascend_baselines import Candidate
 r=json.loads((HERE/'RUNTIME_RECIPE.json').read_text())
 contracts=json.loads((HERE.parent/'ALL_CONTRACTS.json').read_text())
 c=next(x for x in contracts if x['op']=='CausalConv1dUpdateFwdOp' and x['case_id']==case.id)
 assert json.loads(json.dumps(case.params))==c['params']
 transport=DirectACL('CausalConv1dUpdate',torch,r['opapi'],r['nnopbase'])
 def run(x,weight,conv_state,bias):
  assert conv_state.shape[-1]==weight.shape[-1]==4
  packed=x[:,None,:].contiguous()
  # Original Update reads K-1 history rows starting at zero and writes K-1 rows
  # starting at zero (InitRing / WriteBackState). Keep ALL four allocated rows.
  # The canonical window includes one older row which the convolution discards.
  state=torch.roll(conv_state,shifts=-1,dims=-1).transpose(1,2).contiguous()
  assert state.numel()==conv_state.numel()
  updated=conv_state.clone()
  updated[...,0:1].copy_(conv_state[...,1:2])
  out=torch.empty_like(packed)
  transport.call([packed,weight.T.contiguous(),state,None if bias is None else bias.contiguous(),None,None,None,None,None],[out],[(case.params['activation'] or 'none').encode(),-1,1])
  # Copy every API-updated history row; retain the additional canonical row.
  updated[...,1:].copy_(state[:,:3,:].transpose(1,2))
  torch.npu.synchronize()
  run_dir=Path(os.environ['GDN_RUN_DIR'])
  maps=Path('/proc/self/maps').read_text();(run_dir/'loaded.maps').write_text(maps)
  paths=sorted({line.split()[-1] for line in maps.splitlines() if '/' in line and any(s in line for s in ('causalpairs','libnnopbase','libruntime','libascendcl'))})
  loaded=[]
  for path in paths:
   p=Path(path)
   if p.is_file():
    s=p.stat();loaded.append(dict(path=path,realpath=str(p.resolve()),inode=s.st_ino,device=s.st_dev,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
  (run_dir/'NATIVE_LAUNCH.json').write_text(json.dumps(dict(api='aclnnCausalConv1dUpdate',two_stage_call_success=True,synchronized=True,full_api_state_shape=list(state.shape),full_output_shapes=[list(out[:,0,:].shape),list(updated.shape)],loaded=loaded,at=time.time()),indent=2))
  return out[:,0,:],updated
 hashes={x['path']:x['sha256'] for x in json.loads((HERE/'PINS.json').read_text())}
 for n in ['binding.py','direct_acl.py']:hashes[str(HERE/n)]=hashlib.sha256((HERE/n).read_bytes()).hexdigest()
 return Candidate(Implementation(run),'ops-transformer::aclnnCausalConv1dUpdate','Original whole two-stage Update API, full state allocation; transpose/copy K-1 native history rows and retain extra canonical history row. No mathematical replacement.',hashes,timing_mode='host_roundtrip',cleanup=transport.cleanup)
