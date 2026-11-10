import sys,importlib,hashlib,json
from pathlib import Path
sys.dont_write_bytecode=True
w=Path('/opt/tileops-all-pairs/workspace');p=Path(__file__).parent;sys.path[:0]=[str(w/'TileOPs'),str(w/'TileOPs/src'),str(w)]
m=importlib.import_module('benchmarks.ascend_all_pairs.ops_transformer');rows=[]
for op,name in [('GeluTanhAndMulFwdOp','gelu'),('YaRNFwdOp','rope'),('LongRoPEFwdOp','rope'),('FusedTopKFwdOp','topk'),('MoEPostPermuteFwdOp','post'),('GemmINT8W8A8FwdOp','quant'),('MoEPrePermuteFwdOp','pre')]:
 f=m.PROVIDERS[op];assert 'integration700_transformer_dense' in f.__name__;rows.append(dict(op=op,dispatch=f.__name__))
assert 'torch_npu' not in sys.modules
(p/'REGISTRATION_DEVICE_FREE.json').write_text(json.dumps(dict(rows=rows,torch_npu_imported=False),indent=2));print('formal dispatch import passed without torch_npu')
