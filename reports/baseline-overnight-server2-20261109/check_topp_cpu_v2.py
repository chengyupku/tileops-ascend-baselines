exec(__import__('pathlib').Path(__file__).with_name('check_w4_cpu.py').read_text().split('from candidate_w4 import decode')[0])
import candidate_topp_v2 as new
from workloads import sampling as old
rows=[]
for vocab in [1,3,64,257]:
 for kind in ['random','ties','inf','nan']:
  x=torch.randn(4,vocab)
  if kind=='ties':x=torch.floor(x*2)
  if kind=='inf':x[0]=float('-inf');x[1,0]=float('inf')
  if kind=='nan':x[0,0]=float('nan')
  p=torch.tensor([0.,.5,.95,1.]);k=torch.tensor([1,1,max(1,vocab//2),vocab])
  for masked in [False,True]:
   a=old.top_k_mask(x,k) if masked else x
   got=new.top_p_mask(a,p);expected=old.top_p_mask(a,p)
   torch.testing.assert_close(got,expected,rtol=0,atol=0,equal_nan=True)
   rows.append({'vocab':vocab,'kind':kind,'topk_first':masked,'exact_with_nan':True})
(P/'TOPP_CPU_CHECK_v2.json').write_text(json.dumps(rows,indent=2));print('passed',len(rows))
