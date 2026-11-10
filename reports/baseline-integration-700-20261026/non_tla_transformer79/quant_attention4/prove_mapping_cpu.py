"""Index/shape proof only; no torch, NPU, checker or attention computation."""
import json
from pathlib import Path
from adapter import CASES,plan
rows=[]
for case in CASES:
 p=plan(case); b,h,hk,g=p['b'],p['h'],p['hk'],p['groups']
 assert p['expanded_h']<=256 and p['expanded_hk']<=256 and 1<=g<=64
 assert p['expanded_h']//p['expanded_hk']==g
 checks=0; page0=0
 for batch,(nq,nk,npages) in enumerate(zip(p['q_lens'],p['kv_lens'],p['pages'])):
  for head in range(h):
   eh=batch*h+head
   assert eh//g==batch*hk+head//g # original grouping and v-scale row
   for token in range(nk):
    physical=page0+token//128; off=token%128
    assert page0<=physical<page0+npages and 0<=off<128
    assert (physical-page0)*128+off==token
    checks+=1
  # No padded keys participate: original seqused_kv remains nk.
  for qi in range(nq):
   assert 0<=qi+nk-nq<nk # causal right-aligned last visible key unchanged
  page0+=npages
 assert page0==p['kv_shape'][0]
 rows.append(dict(case_id=case,plan=p,index_checks=checks,status='passed',
  limitation='symbolic layout/group/mask equivalence only, no numerical or runtime validation'))
# BF16->FP16 cannot recover original FP16 accuracy, even near one.
# 1+2^-10 is representable in FP16; BF16 rounds it to 1, and casting stays 1.
report=dict(rows=rows,all_passed=True,imports='stdlib and import-safe adapter only',
 dtype_counterexample=dict(real_value=1+2**-10,fp16=1+2**-10,bf16_then_fp16=1.0))
Path(__file__).with_name('CPU_MAPPING_PROOF.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'passed':len(rows),'index_checks':sum(r['index_checks'] for r in rows)}))
