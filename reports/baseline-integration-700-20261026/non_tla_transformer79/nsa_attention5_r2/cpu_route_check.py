"""Standard-library-only structural check; never imports torch or device code."""
import json
from adapter import compressed_plan
cases = {'prefill-4x1k-float16':[1024]*4,
         'prefill-8x1k-float16':[1024]*8,
         'prefill-ragged-bfloat16':[257,767]}
rows=[]
for name,lengths in cases.items():
    offsets,chunks=[0],[0]
    for n in lengths:
        offsets.append(offsets[-1]+n); chunks.append(chunks[-1]+(n+31)//32)
    plan=compressed_plan(offsets,chunks,32)
    actual={}
    for a,b,c,d in plan:
        for t in range(a,b):
            assert t not in actual
            actual[t]=list(range(c,d))
    for i,n in enumerate(lengths):
        for t in range(n):
            expected=list(range(chunks[i],chunks[i]+(t+1)//32))
            assert actual.get(offsets[i]+t,[])==expected
    rows.append({'case':name,'groups':len(plan),'packed_kv':sum(d-c for a,b,c,d in plan),'status':'exact_address_mapping_pass'})
print(json.dumps({'device_imports':0,'checks':rows},indent=2))
