import sys,builtins,json,hashlib,datetime,types
from pathlib import Path
sys.dont_write_bytecode=True
O=Path(__file__).resolve().parent;W=Path('/root/dyq/workspace');orig=builtins.__import__;attempts=[]
def block(n,*a,**kw):
 if n.split('.')[0] in ('torch','torch_npu','acl','tvm','tilelang'):
  attempts.append(n);raise AssertionError('Device import forbidden: '+n)
 return orig(n,*a,**kw)
builtins.__import__=block
sys.path[:0]=[str(W/'TileOPs'),str(W/'TileOPs/src')]
import registration as M
from benchmarks.ascend_all_pairs.ops_transformer import PROVIDERS
p=dict(PROVIDERS);before=dict(PROVIDERS);previous=p['DSADecodeWithKVCacheFwdOp'];M.install(p)
assert p['DSADecodeWithKVCacheFwdOp'].previous is previous and PROVIDERS==before
assert all(p[k] is v for k,v in before.items() if k!='DSADecodeWithKVCacheFwdOp')
files=[W/'TileOPs/benchmarks/ascend_all_pairs/ops_transformer'/n for n in ['__init__.py','adapters.py']]
result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='passed',previous_module=previous.__module__,previous_name=previous.__name__,changed_in_private_dictionary=['DSADecodeWithKVCacheFwdOp'],shared_dictionary_unchanged=True,device_import_attempts=attempts,pins={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},finding='Current public DSA route is adapters.dsa; tested long-context original complete-API route is the distinct frozen integration700_transformer_attention_misc_dsa.py. Proposal selects its unchanged metadata-only variant for two exact keys, preserving adapters.dsa for every other key. No math replacement route is certified.')
(O/'FORMAL_REGISTRY_CHECK.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='pins'},indent=2))
