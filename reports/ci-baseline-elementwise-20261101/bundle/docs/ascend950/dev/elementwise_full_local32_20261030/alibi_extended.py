"""ALiBi CPU references, irregular heads/lengths, fresh outputs and bounded stores."""
import argparse,hashlib,json
from pathlib import Path
import torch,torch_npu,tilelang
from tileops.kernels.ascend.elementwise import alibi as module
from tileops.ops.elementwise import AlibiFwdOp
from workloads.elementwise import ElementwiseWorkload,alibi_reference
from workloads.numerics import compare_outputs
p=argparse.ArgumentParser();p.add_argument('--dtype',required=True);a=p.parse_args();dtype=getattr(torch,a.dtype)
source=Path(module.__file__);frozen=hashlib.sha256(source.read_bytes()).hexdigest();rows=[]
shapes=[(1,1),(1,3),(3,5),(7,3),(31,7),(63,3),(64,8),(65,9),(129,3),(257,5),(513,2),(17,33)]
for seq,heads in shapes:
 op=AlibiFwdOp(seq_len=seq,num_heads=heads,out_dtype=dtype,device='npu')
 contract=ElementwiseWorkload('AlibiFwdOp',(),seq_len=seq,num_heads=heads,out_dtype=dtype,device='cpu').verification()
 ref=alibi_reference(seq,heads,dtype,'cpu');out=op();torch.npu.synchronize();cpu=out.cpu()
 assert out.shape==(heads,seq,seq) and out.dtype==dtype and out.device.type=='npu'
 compare_outputs(cpu,ref,contract);assert torch.equal(cpu,cpu.transpose(1,2))
 diagonal=cpu.diagonal(dim1=1,dim2=2);assert bool((diagonal==0).all()) and bool(torch.signbit(diagonal).all())
 repeated=op();assert repeated.untyped_storage().data_ptr()!=out.untyped_storage().data_ptr();compare_outputs(repeated.cpu(),ref,contract)
 if (seq,heads)==(65,9):
  built=[k for entries in op._built_entries.values() for k in entries.values()];assert len(built)==1
  explicit=tilelang.compile(built[0].compiled.prim_func,target='ascend',out_idx=[])
  count=ref.numel();guard=torch.full((count+128,),321,device='npu',dtype=dtype);target=guard[64:-64];explicit(target);torch.npu.synchronize()
  assert bool((guard[:64]==321).all()) and bool((guard[-64:]==321).all());compare_outputs(target.cpu().reshape(ref.shape),ref,contract)
  bad=target.cpu().reshape(ref.shape);bad[-1,-1,-1]=float('nan')
  try:compare_outputs(bad,ref,contract)
  except AssertionError:pass
  else:raise AssertionError('checker accepted corrupted tail')
 rows.append(dict(seq_len=seq,num_heads=heads,dtype=str(dtype),correct=True,fresh_output=True,symmetric=True,negative_zero_diagonal=True));print(rows[-1],flush=True)
invalid=[]
for seq,heads in [(0,3),(-1,3),(3,0),(3,-1)]:
 try:AlibiFwdOp(seq_len=seq,num_heads=heads,out_dtype=dtype,device='npu')()
 except (ValueError,AssertionError):invalid.append([seq,heads])
 else:raise AssertionError('accepted invalid dimensions')
assert hashlib.sha256(source.read_bytes()).hexdigest()==frozen;torch.npu.synchronize()
Path(__file__).with_name('alibi-extended-'+a.dtype+'.json').write_text(json.dumps(dict(rows=rows,invalid_rejected=invalid,module_sha256=frozen,runtime_limit=torch.npu.get_device_limit(0)),indent=2)+'\n')
