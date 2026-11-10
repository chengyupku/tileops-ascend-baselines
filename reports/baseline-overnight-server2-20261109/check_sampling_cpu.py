exec(__import__('pathlib').Path(__file__).with_name('check_w4_cpu.py').read_text().split('from candidate_w4 import decode')[0])
from workloads import sampling as old
import candidate_sampling as new
rows=[]
for seed in [0,1,2**31+7,2**62+9,-1]:
 for off in [0,2**32+3,-1]:
  a=torch.tensor(seed);b=torch.tensor(off)
  assert torch.equal(old.philox_uniform(a,b,19,9),new.philox_uniform(a,b,19,9))
rows.append({'test':'Philox seeds/offsets','count':15,'exact':True})
for vocab in [1,3,64,257]:
 w=torch.rand(17,vocab);w[:,::3]=0;w[:,-1]=1
 for u in [torch.zeros(17),torch.full((17,),1-2**-24),torch.rand(17)]:
  assert torch.equal(old._draw(w,u),new._draw(w,u))
rows.append({'test':'positive support/zero endpoints','count':12,'exact':True})
for length in [1,2,8]:
 for equal in [True,False]:
  draft=torch.rand(17,length,64);draft=draft/draft.sum(-1,keepdim=True)
  target=torch.cat((draft,draft[:,:1]),1) if equal else torch.softmax(torch.randn(17,length+1,64),-1)
  ids=torch.randint(0,64,(17,length),dtype=torch.int32);args=(draft,ids,target,torch.tensor(7),torch.tensor(0))
  a=old.chain_speculative_sampling(*args);b=new.chain_speculative_sampling(*args)
  assert all(torch.equal(x,y) for x,y in zip(a,b))
rows.append({'test':'chain all-accepted/mixed','count':6,'exact':True})
(P/'SAMPLING_CPU_CHECK.json').write_text(json.dumps(rows,indent=2));print(rows)
