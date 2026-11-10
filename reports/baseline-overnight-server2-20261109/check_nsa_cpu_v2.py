exec(__import__('pathlib').Path(__file__).with_name('check_w4_cpu.py').read_text().split('from candidate_w4 import decode')[0])
import types
from workloads.attention.nsa import NSAFwdWorkload
import candidate_nsa as new
rows=[]
for causal in [False,True]:
 for counts_zero in [False,True]:
  q=torch.randn(7,2,4);k=torch.randn(7,1,4);v=torch.randn(7,1,3)
  index=torch.randint(-1,5,(7,1,2),dtype=torch.int32);counts=torch.zeros(7,1,dtype=torch.int32) if counts_zero else torch.randint(0,3,(7,1),dtype=torch.int32)
  w=types.SimpleNamespace(block_size=2,scale=.7,is_causal=causal,g_slc=torch.rand(1,7,2))
  args=(q,k,v,index,counts,torch.tensor([0,3,7]),None)
  a=NSAFwdWorkload.ref_program(w,*args);b=new.ref_program(w,*args)
  print(causal,counts_zero, "maxabs",float((a-b).abs().max()),"finite",bool(torch.isfinite(a).all()),bool(torch.isfinite(b).all()),flush=True)
  torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)
  rows.append({'causal':causal,'no_blocks':counts_zero,'max_abs':float((a-b).abs().max())})
(P/'NSA_CPU_CHECK.json').write_text(json.dumps(rows,indent=2));print('passed',len(rows))

for n,heads,slots,d in [(7,2,4,3),(5,4,6,7)]:
 x=torch.randn(n,heads,d);pos=torch.randint(0,n,(n,slots,heads));head=torch.arange(heads)
 a=x[pos,head];flat=(pos*heads+head).flatten();b=x.reshape(-1,d).index_select(0,flat).reshape(*pos.shape,d)
 assert torch.equal(a,b)
print('gather values exactly equal; subsequent einsum may select different layout path')
