exec(__import__('pathlib').Path(__file__).with_name('check_w4_cpu.py').read_text().split('from candidate_w4 import decode')[0])
from adapters import gated_fp32_v7 as old
import candidate_gated_v2 as new
rows=[]
for kind in ['gdn','kda']:
 for length in [1,5,19,65]:
  for packed in [False,True]:
   b=1 if packed else 2;h=4;kdim=4;vdim=3
   q=torch.randn(b,length,2,kdim);k=torch.randn_like(q)*.1;v=torch.randn(b,length,h,vdim)
   g=-torch.rand((b,length,h,kdim) if kind=='kda' else (b,length,h));beta=torch.rand(b,length,h)
   bounds=torch.tensor([0,0,length//2,length],dtype=torch.long) if packed else None
   initial=torch.randn(3 if packed else b,h,kdim,vdim)*.1
   fn0=getattr(old,kind+'_ref');fn1=getattr(new,kind+'_ref')
   args=(q,k,v,g,beta,initial,bounds,bounds)
   a=fn0(*args);c=fn1(*args)
   for x,y in zip(a,c):torch.testing.assert_close(x,y,atol=1e-5,rtol=1e-5)
   rows.append({'kind':kind,'length':length,'packed_with_empty_sequence':packed,'max_abs':[float((x-y).abs().max()) for x,y in zip(a,c)]})
(P/'GATED_SOLVE_CPU_CHECK.json').write_text(json.dumps(rows,indent=2));print('passed',len(rows))
