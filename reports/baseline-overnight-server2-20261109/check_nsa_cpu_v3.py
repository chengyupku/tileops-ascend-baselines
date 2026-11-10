exec(__import__('pathlib').Path(__file__).with_name('check_w4_cpu.py').read_text().split('from candidate_w4 import decode')[0])
import types
from workloads.attention.nsa import NSAFwdWorkload
import candidate_nsa_v2 as new
rows=[]
for causal in [False,True]:
 for counts_zero in [False,True]:
  q=torch.randn(7,2,4);k=torch.randn(7,1,4);v=torch.randn(7,1,3)
  index=torch.randint(-1,5,(7,1,2),dtype=torch.int32);counts=torch.zeros(7,1,dtype=torch.int32) if counts_zero else torch.randint(0,3,(7,1),dtype=torch.int32)
  w=types.SimpleNamespace(block_size=2,scale=.7,is_causal=causal,g_slc=torch.rand(1,7,2))
  args=(q,k,v,index,counts,torch.tensor([0,3,7]),None)
  a=NSAFwdWorkload.ref_program(w,*args);b=new.ref_program(w,*args)
  assert torch.equal(a,b)
  rows.append({'causal':causal,'no_blocks':counts_zero,'all_values_bit_exact':True})
(P/'NSA_CPU_CHECK_v3.json').write_text(json.dumps(rows,indent=2));print('passed',len(rows))
