import sys,pathlib,json,copy,builtins,types,hashlib
sys.dont_write_bytecode=True
O=pathlib.Path(__file__).resolve().parent
real_import=builtins.__import__
def safe(name,*a,**kw):
 if name.split('.')[0] in {'torch','torch_npu','tilelang','tvm','ctypes'}:raise AssertionError('Forbidden runtime import '+name)
 return real_import(name,*a,**kw)
builtins.__import__=safe
sys.path[:0]=[str(O),str(O.parents[2]/'TileOPs')]
import registration as r
from build_contracts import cases
from benchmarks.ascend_baselines import UnsupportedCase
checks=[]
def ok(n):checks.append(n)
def raises(fn,cls):
 try:fn()
 except cls:return
 raise AssertionError('Expected '+cls.__name__)
cs=cases();assert len(cs)==3
for c in cs:
 assert r.matches(c)
 candidate=r.make_candidate(c);assert not any(n in sys.modules for n in ['torch','torch_npu','tilelang','tvm'])
 raises(lambda:candidate.implementation.run(),RuntimeError)
 raises(lambda:candidate.implementation.run(),RuntimeError)
 candidate.cleanup()
 for k in c._call.params:
  bad=copy.deepcopy(c);bad._call.params[k]='deliberately_invalid'
  raises(lambda:r.make_candidate(bad),UnsupportedCase)
 for name in c._call.signature.inputs:
  bad=copy.deepcopy(c);bad._call.specs[name]=None if bad._call.specs.get(name) else types.SimpleNamespace(shape=(1,),dtype='float32',values=None,cpu=False)
  raises(lambda:r.make_candidate(bad),UnsupportedCase)
 args=[types.SimpleNamespace(shape=s.shape,dtype='torch.'+s.dtype,device='cpu' if s.cpu else 'npu:0') for n in c._call.signature.inputs if c._call.present(n) for s in [c._call.specs[n]]]
 r.validate_inputs(c,args,{})
 raises(lambda:r.validate_inputs(c,args+[None],{}),UnsupportedCase)
ok('three actual manifest contracts, all scalar params and optional inputs fail closed; present-only input arity')
# Read actual provider dictionary with runtime import blacklist active; mutate a copy only.
from benchmarks.ascend_all_pairs import tilelang as provider
before=dict(provider.PROVIDERS);target=dict(before);r.install(target)
assert provider.PROVIDERS==before
for c in cs:assert target[c._call.signature.name](c).api.startswith('tilelang::')
unknown=copy.deepcopy(cs[0]);object.__setattr__(unknown,'id','outside-scope')
sentinel=object();assert r.wrap(lambda c:sentinel)(unknown) is sentinel
bad=copy.deepcopy(cs[0]);bad._call.params['is_causal']=False
raises(lambda:r.wrap(lambda c:sentinel)(bad),UnsupportedCase)
ok('actual TILELANG provider dictionary copy; unknown delegated, known mismatch public UnsupportedCase')
raw=r.snapshot();assert hashlib.sha256(raw[r.ADAPTER]).hexdigest()=='fdf4bc496ac2c06b2cb6290680652b77248a95ba17c443561f3eedcc86f77500'
raises(lambda:r.checked(O/'SCOPE.json','0'*64),RuntimeError)
ok('exact adapter bytes and corrupted dependency rejection')
# Simulated lifecycle control flow, without any native import or fake numerical claim.
r.environment=lambda:('env','/private');r.require_guard=lambda:('guard',)
r.validate_inputs=lambda *a:None;r.loaded_libraries=lambda:None
admissions=[];r.admit=lambda runtime=True:(admissions.append(runtime) or raw,('guard',),('env','/private'))
r.tempfile.mkdtemp=lambda **k:'/private/candidate'
clean=[]
inner=types.SimpleNamespace(implementation=types.SimpleNamespace(run=lambda *a:'sentinel'),cleanup=lambda:clean.append(1))
r.private_load=lambda data:lambda path:types.SimpleNamespace(factory=lambda *a:inner)
# cleanup's real torch synchronize is intentionally not executed; guard mismatch tested before it.
c=r.make_candidate(cs[0]);assert c.implementation.run()=='sentinel';assert c.implementation.run()=='sentinel';assert admissions==[True,False]
r.require_guard=lambda:('different',)
raises(c.cleanup,RuntimeError);raises(lambda:c.implementation.run(),RuntimeError)
ok('mocked two-call lifecycle: full runtime hash once, guard crossing rejected before native cleanup')
assert not any(n in sys.modules for n in ['torch','torch_npu','tilelang','tvm'])
(O/'CPU_CHECK.json').write_text(json.dumps({'passed':checks,'device_imports':0,'npu_calls':0,'accepted_added':0,'limits':'lifecycle mocks are control-flow checks only; real runtime cleanup and numerical validation not executed'},indent=2)+'\n')
print(json.dumps(checks))
