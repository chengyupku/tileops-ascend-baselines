"""Original GQA formula with explicit CPU-equivalent saturating FP8 cast."""
import inspect,textwrap,torch
from workloads.attention.gqa.prefill_paged_kv_append import GQAPrefillPagedWithKVCacheFwdWorkload

def bind(workload):
 source=textwrap.dedent(inspect.getsource(GQAPrefillPagedWithKVCacheFwdWorkload.ref_program))
 for name in ('k','v'):
  old=f'({name}_b.float() / {name}_scale[0]).to({name}_pages.dtype)'
  new=f'({name}_b.float() / {name}_scale[0]).clamp(-torch.finfo({name}_pages.dtype).max, torch.finfo({name}_pages.dtype).max).to({name}_pages.dtype)'
  assert source.count(old)==1
  source=source.replace(old,new)
 scope={'torch':torch};exec(compile(source,__file__,'exec'),scope)
 return lambda *args:scope['ref_program'](workload,*args)
