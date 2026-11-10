"""Exercise retirement and precision gates without importing a device runtime."""
import ast
import builtins
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

ROOT=Path('/root/dyq/workspace')
OUT=Path(__file__).resolve().parent
checks=[]
class UnsupportedCase(Exception):pass

def forbidden(*args,**kwargs):
    raise AssertionError('Device/compiler runtime must not be touched by a rejected route')

def imports(name, globals=None, locals=None, fromlist=(), level=0):
    if name=='torch':return SimpleNamespace(bfloat16='bfloat16',float16='float16')
    if name=='torch.nn.functional':return SimpleNamespace(nn=SimpleNamespace(functional=SimpleNamespace()))
    if name=='benchmarks.api':return SimpleNamespace(Implementation=forbidden)
    if name=='benchmarks.ascend_baselines':return SimpleNamespace(UnsupportedCase=UnsupportedCase,WORKSPACE=ROOT)
    if name.startswith(('torch','tilelang')):raise AssertionError('Unexpected runtime import: '+name)
    return builtins.__import__(name,globals,locals,fromlist,level)

def function(path,name):
    tree=ast.parse((ROOT/path).read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    namespace={'__builtins__':dict(vars(builtins),__import__=imports),'UnsupportedCase':UnsupportedCase,'_runtime':forbidden}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),path,'exec'),namespace)
    return namespace[name]

fn=function('TileOPs/benchmarks/ascend_tilelang_attention_cases.py','tilelang_gqa_cases')
try:fn(SimpleNamespace(inputs=[SimpleNamespace(dtype='float16')]*3))
except UnsupportedCase as e:assert 'internal precision' in str(e)
else:raise AssertionError('FP16 source rewriting remained enabled')
checks.append('FP16 rejected before runtime/compiler initialization')
for relative,name in [('truth.py','extrema'),('truth.py','count_nonzero'),('reduce.py','extrema')]:
    try:function('TileOPs/benchmarks/ascend_all_pairs/tilelang/'+relative,name)(None)
    except UnsupportedCase as e:assert 'retired' in str(e)
    else:raise AssertionError('Retired algorithm is still callable')
    checks.append(relative+':'+name+' rejects before any device import')
source=(ROOT/'TileOPs/benchmarks/ascend_tilelang_attention_cases.py').read_text()
assert 'adapted = source\n' in source and 'DtypeOnly' not in source and 'source.replace' not in source
checks.append('Attention source copies preserve original internal precision')
hashes=json.loads((ROOT/'reports/baseline-source-integrity-20261027/EVIDENCE_HASHES.json').read_text())
for suffix in ['TileOPs/benchmarks/ascend_tilelang_generic.py','TileOPs/benchmarks/ascend_all_pairs/fla/normalization.py','TileOPs/benchmarks/ascend_all_pairs/sgl/composed.py','TileOPs/benchmarks/ascend_all_pairs/ops_transformer/composed_activations.py','TileOPs/benchmarks/ascend_all_pairs/tilelang/attention.py']:
    p=ROOT/suffix
    assert hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p)],suffix
checks.append('Optional-autotune path, other compositions and canonical BF16 provider unchanged')
summary=json.loads((ROOT/'.worktrees/TileOPs-baseline-catalog/docs/ascend950/baselines/summary.json').read_text())
assert summary['status_counts']['verified']==2930
assert summary['status_counts']['excluded']==4061
assert summary['integration700_verified']==477
checks.append('Catalog counts changed only by six retired verified pairs; integration700 unchanged')
(OUT/'CPU_CHECKS.json').write_text(json.dumps({'passed':len(checks),'checks':checks,'npu_runs':0},ensure_ascii=False,indent=2)+'\n')
print('\n'.join(checks))
