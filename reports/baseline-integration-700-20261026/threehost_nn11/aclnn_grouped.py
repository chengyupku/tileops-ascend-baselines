"""Original complete aclnnMm API, FP32 lossless widening and group/layout binding."""
import ctypes as C
from pathlib import Path
import json,hashlib,os,time

def make(case):
 import torch
 from benchmarks.api import Implementation
 from benchmarks.ascend_baselines import Candidate,LIBRARIES,source_hashes
 from benchmarks.ascend_math_more import _MathCall
 assert case._call.signature.name=='GroupedGemmFwdOp'
 b=_MathCall('aclnnMm',[C.c_void_p]*3+[C.c_int8],library='ops-nn');outdir=Path.cwd()
 for name,func in [('aclnnMmGetWorkspaceSize',b.size),('aclnnMm',b.launch)]:
  address=C.cast(func,C.c_void_p).value;matches=[]
  for line in Path('/proc/self/maps').read_text().splitlines():
   a=line.split();lo,hi=[int(x,16) for x in a[0].split('-')]
   if lo<=address<hi:
    f=Path(a[-1]);s=f.stat();matches.append(dict(line=line,path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),inode=s.st_ino,device=s.st_dev))
  with (outdir/'ACTUAL_ACLNN_SYMBOLS.jsonl').open('a') as f:f.write(json.dumps(dict(symbol=name,pointer=address,maps=matches))+'\n')
 # Preserve original helper behavior; record return codes of both original APIs.
 for attr in ['size','launch']:
  original=getattr(b,attr)
  def record(*args,_original=original,_attr=attr):
   t=time.time();rc=_original(*args)
   with (outdir/'ACLNN_CALLS.jsonl').open('a') as f:f.write(json.dumps(dict(api=_attr,returncode=rc,elapsed_s=time.time()-t))+'\n')
   return rc
  setattr(b,attr,record)
 def mm(a,w):
  aa=a.float().contiguous();ww=w.float().contiguous();y=torch.empty((aa.shape[0],ww.shape[1]),device=a.device,dtype=torch.float32);r=b.record()
  b.call([b.tensor(aa,r),b.tensor(ww,r),b.tensor(y,r),0],r,a.device)
  return y.to(a.dtype)
 def run(a,w,sizes,offsets):
  pieces=[];start=0
  for i,n in enumerate(sizes.tolist()):
   pieces.append(mm(a[start:start+n],w[i].T if case.params['transpose_b'] else w[i]));start+=n
  return torch.cat(pieces)
 return Candidate(Implementation(run),'ops-nn::aclnnMm (original MatMulV3 full API)','Original API, exact FP32 widening and layout only; retain complete output and all group segments.',b.hashes(LIBRARIES/'ops-nn/matmul/mat_mul_v3')|source_hashes(Path(__file__)),cleanup=b.cleanup)
