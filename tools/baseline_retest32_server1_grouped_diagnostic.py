"""Diagnostic only: unchanged checker with side observations, never cached."""
import sys,json,dataclasses,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'reports/baseline-cache-20261027/python'))
a=sys.argv[1:];out=Path(a[a.index('--out')+1]);task=json.loads(Path(a[a.index('--task')+1]).read_text())
from benchmarks.ascend_all_pairs import registry
original=registry.candidate_factory
seen=[]
def factory(case,library):
 make=original(case,library)
 def build():
  c=make();impl=c.implementation
  def run(*inputs):
   result=impl.run(*inputs)
   if not seen:
    seen.append(True)
    import torch
    torch.npu.synchronize()
    n=min(64,int(inputs[2][0].item()));a=inputs[0][:n].cpu();b=inputs[1][0].cpu()
    if case.params['transpose_b']:b=b.T.contiguous()
    actual=result[:n].cpu();fp64=(a.double()@b.double());rounded=fp64.to(actual.dtype)
    refs={'default_cpu_bf16':a@b,'fp64_rounded':rounded}
    if hasattr(torch.backends,'mkldnn'):
     with torch.backends.mkldnn.flags(enabled=False):refs['cpu_bf16_mkldnn_disabled']=a@b
    data={'shapes':[list(a.shape),list(b.shape)],'cpu_capability':torch.backends.cpu.get_cpu_capability(),'mkldnn_available':torch.backends.mkldnn.is_available(),'torch_version':torch.__version__,'comparisons':{}}
    for name,r in refs.items():
     data['comparisons'][name]={'vs_candidate_not_close':int((~torch.isclose(actual.double(),r.double(),rtol=.016,atol=.016)).sum()),'vs_fp64_rounded_not_close':int((~torch.isclose(rounded.double(),r.double(),rtol=.016,atol=.016)).sum()),'candidate_max_abs':float((actual.double()-r.double()).abs().max()),'exact_matches_candidate':int((actual==r).sum()),'elements':r.numel()}
    (out/'cpu_reference_diagnostic.json').write_text(json.dumps(data,indent=2));print(json.dumps(data),flush=True)
   return result
  return dataclasses.replace(c,implementation=dataclasses.replace(impl,run=run))
 return build
registry.candidate_factory=factory
p=ROOT/'TileOPs/scripts/cache_frozen_pair_check.py';spec=importlib.util.spec_from_file_location('original_checker_diagnostic',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
sys.argv=[str(p),'--op',task['key'][1],'--case-id',task['key'][2],'--library',task['key'][0],'--output',str(out/'correctness.json')]
raise SystemExit(m.main())
