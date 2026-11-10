"""Sinusoidal CPU references, irregular dims/lengths, fresh outputs and bounded stores."""
import argparse,hashlib,json
from pathlib import Path
import torch,torch_npu,tilelang
from tileops.kernels.ascend.elementwise import sinusoidal as module
from tileops.ops.elementwise import SinusoidalFwdOp
from workloads.elementwise import ElementwiseWorkload,sinusoidal_reference
from workloads.numerics import compare_outputs
p=argparse.ArgumentParser();p.add_argument('--dtype',required=True);a=p.parse_args();dtype=getattr(torch,a.dtype)
source=Path(module.__file__);frozen=hashlib.sha256(source.read_bytes()).hexdigest();rows=[]
shapes=[(1,2),(1,6),(3,10),(7,6),(31,14),(63,126),(64,128),(65,130),(129,514),(257,1026),(513,258),(17,66)]
for seq,dims in shapes:
 op=SinusoidalFwdOp(seq_len=seq,d_model=dims,out_dtype=dtype,device='npu')
 contract=ElementwiseWorkload('SinusoidalFwdOp',(),seq_len=seq,d_model=dims,out_dtype=dtype,device='cpu').verification()
 ref=sinusoidal_reference(seq,dims,dtype,'cpu');out=op();torch.npu.synchronize();cpu=out.cpu()
 assert out.shape==(seq,dims) and out.dtype==dtype and out.device.type=='npu'
 compare_outputs(cpu,ref,contract)
 assert bool((cpu[0,0::2]==0).all()) and bool((cpu[0,1::2]==1).all())
 internal=[k for entries in op._built_entries.values() for k in entries.values()];assert len(internal)==1
 freq_before=internal[0].frequencies.cpu().clone()
 repeated=op();assert repeated.untyped_storage().data_ptr()!=out.untyped_storage().data_ptr();compare_outputs(repeated.cpu(),ref,contract)
 if (seq,dims)==(65,130):
  built=[k for entries in op._built_entries.values() for k in entries.values()];assert len(built)==1
  explicit=tilelang.compile(built[0].compiled.prim_func,target='ascend',out_idx=[])
  count=ref.numel();guard=torch.full((count+128,),321,device='npu',dtype=dtype);target=guard[64:-64].reshape(ref.shape);explicit(built[0].frequencies,target);torch.npu.synchronize()
  assert bool((guard[:64]==321).all()) and bool((guard[-64:]==321).all());compare_outputs(target.cpu().reshape(ref.shape),ref,contract)
  bad=target.cpu().reshape(ref.shape);bad[-1,-1]=float('nan')
  try:compare_outputs(bad,ref,contract)
  except AssertionError:pass
  else:raise AssertionError('checker accepted corrupted tail')
 assert torch.equal(freq_before,internal[0].frequencies.cpu())
 rows.append(dict(seq_len=seq,d_model=dims,dtype=str(dtype),correct=True,fresh_output=True,first_row_exact=True,frequencies_unchanged=True));print(rows[-1],flush=True)
invalid=[]
for seq,dims in [(0,2),(-1,2),(3,0),(3,-2),(3,3)]:
 try:SinusoidalFwdOp(seq_len=seq,d_model=dims,out_dtype=dtype,device='npu')()
 except (ValueError,AssertionError):invalid.append([seq,dims])
 else:raise AssertionError('accepted invalid dimensions')
assert hashlib.sha256(source.read_bytes()).hexdigest()==frozen;torch.npu.synchronize()
Path(__file__).with_name('sinusoidal-extended-'+a.dtype+'.json').write_text(json.dumps(dict(rows=rows,invalid_rejected=invalid,module_sha256=frozen,runtime_limit=torch.npu.get_device_limit(0)),indent=2)+'\n')
