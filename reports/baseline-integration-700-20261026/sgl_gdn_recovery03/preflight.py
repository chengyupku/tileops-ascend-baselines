import ast,sys,types,json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent; W=R.parents[2]; sys.path[:0]=[str(W/'TileOPs'),str(R)]
# Load actual registry (stdlib only) and stub only runtime-heavy provider package.
from benchmarks.ascend_all_pairs import registry
sgl=types.ModuleType('benchmarks.ascend_all_pairs.sgl'); sgl.PROVIDERS={}; sys.modules[sgl.__name__]=sgl
from route_install import install
calls=[]
def expected(case): calls.append(case._call.signature.name); return 'SENTINEL'
install(types.SimpleNamespace(PROVIDERS={'GDNFwdOp':expected,'DeltaNetFwdOp':expected}))
for op in ['GDNFwdOp','DeltaNetFwdOp']:
 case=types.SimpleNamespace(_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=op)))
 assert registry.candidate_factory(case,'sgl-kernel-npu')()=='SENTINEL'
checker=W/'TileOPs/scripts/ascend_pair_check.py'; tree=ast.parse(checker.read_text())
assert any(isinstance(n,ast.ImportFrom) and n.module=='benchmarks.ascend_all_pairs.registry' and any(x.name=='candidate_factory' for x in n.names) for n in ast.walk(tree))
(R/'ROUTE_PREFLIGHT.json').write_text(json.dumps({'passed':True,'actual_registry':registry.__file__,'invoked_factories':calls,'checker_sha256':hashlib.sha256(checker.read_bytes()).hexdigest(),'method':'Actual registry candidate_factory executed with runtime package stub; same install() as device runner; checker import AST verified'},indent=2))
