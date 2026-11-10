import ast,sys,types,json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent; W=R.parents[2]; sys.path[:0]=[str(W/'TileOPs'),str(R)]
# Load actual registry (stdlib only) and stub only runtime-heavy provider package.
from benchmarks.ascend_all_pairs import registry
sgl=types.ModuleType('benchmarks.ascend_all_pairs.sgl'); sgl.PROVIDERS={}; sys.modules[sgl.__name__]=sgl
from route_install import install
calls=[]
def expected(case): calls.append(case._call.signature.name); return 'SENTINEL'
install(types.SimpleNamespace(PROVIDERS={'MLAKVCacheWriteFwdOp':expected}))
for op in ['MLAKVCacheWriteFwdOp']:
 case=types.SimpleNamespace(_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=op)))
 assert registry.candidate_factory(case,'sgl-kernel-npu')()=='SENTINEL'
checker=W/'TileOPs/scripts/ascend_pair_check.py'; tree=ast.parse(checker.read_text())
assert any(isinstance(n,ast.ImportFrom) and n.module=='benchmarks.ascend_all_pairs.registry' and any(x.name=='candidate_factory' for x in n.names) for n in ast.walk(tree))
(R/'ROUTE_PREFLIGHT.json').write_text(json.dumps({'passed':True,'actual_registry':registry.__file__,'invoked_factories':calls,'checker_sha256':hashlib.sha256(checker.read_bytes()).hexdigest(),'method':'Actual registry candidate_factory executed with runtime package stub; same install() as device runner; checker import AST verified'},indent=2))
# Execute the actual private factory with runtime-only imports stubbed.
from benchmarks import ascend_baselines as ab
sys.modules['torch']=types.ModuleType('torch')
a=types.ModuleType('benchmarks.api'); a.Implementation=lambda fn:types.SimpleNamespace(run=fn);sys.modules[a.__name__]=a
native_calls=[]
def native(*args): native_calls.append(args);return lambda *a,**k:None
ab.local_function=native
import integration700_sgl_attention32 as private
install(private)
tensor=types.SimpleNamespace(dtype='bf16',shape=(64,512))
case=types.SimpleNamespace(_call=types.SimpleNamespace(signature=types.SimpleNamespace(name='MLAKVCacheWriteFwdOp')), inputs=(tensor,tensor,tensor),params={'fuse_rope':False})
candidate=registry.candidate_factory(case,'sgl-kernel-npu')()
assert candidate.implementation.run.__module__==private.__name__
assert any('kv_cache_store' in str(x) for x in native_calls)
assert any('integration700_sgl_attention32.py' in x for x in candidate.source_sha256)
(R/'PRIVATE_FACTORY_PREFLIGHT.json').write_text(json.dumps({'passed':True,'factory':private.write.__module__+'.write','implementation':candidate.implementation.run.__qualname__,'api':candidate.api,'source_sha256':candidate.source_sha256,'native_resolutions':native_calls,'torch_imported':False},indent=2))
