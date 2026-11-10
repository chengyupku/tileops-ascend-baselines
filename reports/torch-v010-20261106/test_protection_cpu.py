import os,sys,pathlib,json
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
P=pathlib.Path(__file__).resolve().parent
sys.path[:0]=[str(P/'source/src'),str(P/'source'),str(P)]
import torch
from workloads.attention.mla_cache_write import MLAKVCacheWriteWorkload
from adapters.protection_reference import remove_protection_clones
from adapters.execution_boundary import private_inputs
from types import SimpleNamespace
# The non-RoPE branch accesses no instance fields.
ref=MLAKVCacheWriteWorkload.ref_program.__get__(SimpleNamespace(),SimpleNamespace)
fn,proof=remove_protection_clones(ref,['kv_cache'])
args=(torch.randn(3,4),torch.randn(3,2),torch.randn(2,4,6),torch.tensor([0,-1,6]))
expected=ref(*args);original=args[2].clone();b=private_inputs(fn,args,(2,))
for _ in range(3):assert torch.equal(b.check_once(lambda:None),expected)
assert torch.equal(args[2],original)
assert torch.equal(expected.view(-1,6)[1],original.view(-1,6)[1])
(P/'PROTECTION_CPU_TEST.json').write_text(json.dumps({'MLAKVCacheWrite':True,'repeat_reset':True,'unwritten_preserved':True,'original_cache_preserved':True,'proof':proof},indent=2));print('passed')
