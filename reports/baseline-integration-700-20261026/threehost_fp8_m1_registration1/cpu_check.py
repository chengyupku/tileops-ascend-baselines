"""CPU only: real manifest/Case, hostile imports, new route/lifetime behavior."""
import ast,builtins,copy,dataclasses,functools,hashlib,json,sys,tempfile,types,shutil
from pathlib import Path
from unittest.mock import patch
P=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path[:0]=[str(P),'/root/dyq/workspace/TileOPs','/root/dyq/workspace/TileOPs/src']
original=builtins.__import__
def guarded_import(name,*args,**kwargs):
    if name.split('.')[0] in ('torch','torch_npu','acl','ctypes'):
        raise AssertionError('Forbidden CPU import: '+name)
    return original(name,*args,**kwargs)
builtins.__import__=guarded_import
from tileops.manifest import load_manifest,load_adts,load_workloads
from tileops.manifest.plan import entry_plan
from tileops.manifest.workload import instantiate,Call
from tileops.manifest.values import ADTValue
from typing import Any,Callable,Optional
from benchmarks.ascend_baselines import UnsupportedCase
import integration700_threehost_fp8_m1_registration1 as m
nodes=[n for n in ast.parse((P/'frozen/api.py').read_text()).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ('Case','_report_value')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'original Case AST','exec'))
plan=entry_plan(m.KEY[1],load_manifest()[m.KEY[1]],load_adts(),resolve=False)
call=next(c for r in load_workloads(m.KEY[1]) for d in r.get('dtype_cases') or [{}] if (c:=instantiate(plan,r,d)).case_id==m.KEY[2])
case=Case(call.case_id,call,None)
checks=[]
def ok(n):checks.append(n)
def rejected(fn,exc=UnsupportedCase):
    try:fn()
    except exc:return
    raise AssertionError('Expected '+exc.__name__)
assert m.matches(case)
route=m.wrap(lambda c:'previous')
candidate=route(case)
assert candidate.cleanup and callable(candidate.implementation.run)
assert not any(n in sys.modules for n in ['torch','torch_npu','acl','ctypes'])
ok('real manifest Call and original Case route; CPU lazy construction under forbidden-import hook')
unknown=Case('unknown',call,None)
assert route(unknown)=='previous'
ok('unknown key delegates previous')
for name in ['a','b','scale_a','scale_b','d']:
    for field,value in [('shape',(2,3)),('dtype','float16')]:
        c=copy.deepcopy(call);c.specs[name]=dataclasses.replace(c.specs[name],**{field:value})
        rejected(lambda:route(Case(c.case_id,c,None)))
c=copy.deepcopy(call);c.specs['bias']=dataclasses.replace(c.specs['d'],shape=(7168,))
rejected(lambda:route(Case(c.case_id,c,None)))
c=copy.deepcopy(call);c.params['out_dtype']='float16'
rejected(lambda:route(Case(c.case_id,c,None)))
c=copy.deepcopy(call);c.indices['M']=2
rejected(lambda:route(Case(c.case_id,c,None)))
c=copy.deepcopy(call);c.signature.inputs['a']=dataclasses.replace(c.signature.inputs['a'],mutated=True)
rejected(lambda:route(Case(c.case_id,c,None)))
ok('same key rejects shapes/dtypes/scales/bias/output/indices/signature; public UnsupportedCase')
class Device:
    type='npu'
    def __str__(self):return 'npu:0'
def tensor(shape,dtype):return types.SimpleNamespace(shape=shape,dtype='torch.'+dtype,layout='torch.strided',device=Device())
args=[tensor(s,d) for s,d in [((1,2048),'float8_e4m3fn'),((7168,2048),'float8_e4m3fn'),((1,16),'float32'),((7168,16),'float32')]]
m._inputs(args,{})
for bad in [args[:3],args+[None]]:rejected(lambda:m._inputs(bad,{}))
rejected(lambda:m._inputs(args,{'bias':None}))
for i in range(4):
    bad=copy.deepcopy(args);bad[i].shape=(9,9);rejected(lambda:m._inputs(bad,{}))
    bad=copy.deepcopy(args);bad[i].dtype='torch.float16';rejected(lambda:m._inputs(bad,{}))
ok('invocation tuple arity, keyword bias, each input shape/dtype validated before host/device')
rejected(lambda:candidate.implementation.run(*args),RuntimeError)
candidate.cleanup();candidate.cleanup()
rejected(lambda:candidate.implementation.run(*args),RuntimeError)
ok('wrong host fails closed; unbuilt cleanup idempotent')
with patch.object(m,'_host'),patch.object(m,'_guard',side_effect=RuntimeError('unguarded')),patch.object(m,'_build_child') as build:
    rejected(lambda:m.make(case).implementation.run(*args),RuntimeError)
    assert build.call_count==0
ok('missing guard refuses before native child construction')
# Exercise lifecycle only with a fake child; no native/math execution or claimed numeric pass.
events=[]
child=types.SimpleNamespace(implementation=types.SimpleNamespace(run=lambda *a:tensor((1,7168),'bfloat16')),cleanup=lambda:events.append('cleanup'))
with patch.object(m,'_host'),patch.object(m,'_guard',return_value=('guard',)),patch.object(m,'_environment'),patch.object(m,'_runtime_pins',return_value={}),patch.object(m,'_maps'),patch.object(m,'_build_child',return_value=child) as build:
    c=m.make(case);c.implementation.run(*args);c.implementation.run(*args);assert build.call_count==1
    c.cleanup();c.cleanup();assert events==['cleanup']
    c=m.make(case)
    child.implementation.run=lambda *a:(_ for _ in ()).throw(ValueError('native failure'))
    rejected(lambda:c.implementation.run(*args),ValueError)
    assert events==['cleanup','cleanup']
ok('mock child builds once; original cleanup called once after success or run failure')
# Dependency bytes are verified, not imported. Deliberate mutation stays in a private temp directory.
with tempfile.TemporaryDirectory(dir=P) as d:
    p=Path(d)/'dependency';p.write_bytes(b'original');h=hashlib.sha256(p.read_bytes()).hexdigest()
    m._checked(p,h);p.write_bytes(b'changed');rejected(lambda:m._checked(p,h),RuntimeError)
    data={'RUNTIME_PINS.json':json.dumps({'source':{str(p):h}}).encode()}
    rejected(lambda:m._runtime_pins(data),RuntimeError)
ok('local and runtime dependency tampering fail closed before native import')
with tempfile.TemporaryDirectory(dir=P) as d:
    root=Path(d)
    shutil.copytree(P/'frozen',root/'frozen')
    for name in ('LOCAL_PINS.json','SOURCE_PINS.json','RUNTIME_PINS.json','EXACT_CONTRACT.json'):
        shutil.copyfile(P/name,root/name)
    with patch.object(m,'ROOT',root):
        assert m.matches(case)
        f=root/'frozen/binding.py';f.write_bytes(f.read_bytes()+b'\n# tampered\n')
        rejected(lambda:m.make(case),RuntimeError)
ok('real factory rejects tampered frozen binding through pinned manifest')
old=P.parent/'threehost_dense48_r2/binding.py'
assert (P/'frozen/binding.py').read_bytes()==old.read_bytes()
for filename,names in [('binding.py',['build']),('ascend_math_more.py',['_MathCall']),('ascend_baselines.py',['_owned_native'])]:
    src=(P/'frozen'/filename).read_bytes()
    assert hashlib.sha256(src).hexdigest() in json.loads((P/'SOURCE_PINS.json').read_text()).values()
    tree=ast.parse(src);assert all(any(getattr(n,'name',None)==v for n in tree.body) for v in names)
ok('binding bytes unchanged; complete _MathCall/_owned_native files match accepted frozen hashes')
# Verify the diagnostic destination without executing binding code or reading guard state.
with tempfile.TemporaryDirectory(dir=P) as d:
    root=Path(d);fake_lock=root/'guard';fake_lock.write_text('{}')
    _,data=m._metadata();seen={}
    def load(data,directory,pins):
        seen['directory']=directory
        return types.SimpleNamespace(build=lambda case:child)
    original_read=Path.read_text
    def read(path,*a,**kw):
        if str(path)=='/run/lock/ascend-npu/device-0.lock':return '{}'
        return original_read(path,*a,**kw)
    with patch.object(m,'RUN_ROOT',root/'runs'),patch.object(m,'_load',side_effect=load),patch.object(Path,'read_text',read):
        m._build_child(data,{})
    dest=seen['directory'];assert dest.parent==root/'runs'
    assert (dest/'binding.py').read_bytes()==old.read_bytes()
    assert (dest/'vendor').resolve()==m.DEPLOYMENT/'vendor'
    assert not (dest/'native.json').exists()
ok('private unique __file__/vendor alias isolates native.json from accepted evidence')
# Exercise the real private loader with an inert ctypes token, never importing ctypes
# or invoking build. This checks module wiring despite a hostile ambient namesake.
with tempfile.TemporaryDirectory(dir=P) as d:
    dummy=types.ModuleType('_inert_ctypes_token')
    def inert_import(name,*a,**kw):
        if name=='ctypes':return dummy
        return guarded_import(name,*a,**kw)
    _,data=m._metadata()
    hostile=types.ModuleType('benchmarks.ascend_math_more')
    hostile._MathCall=object()
    with patch.object(builtins,'__import__',inert_import),patch.dict(sys.modules,{'benchmarks.ascend_math_more':hostile}):
        b=m._load(data,Path(d),json.loads(data['RUNTIME_PINS.json']))
    importer=b.build.__globals__['__builtins__']['__import__']
    math=importer('benchmarks.ascend_math_more',fromlist=('_MathCall',))
    assert math is not hostile
    owned=math._MathCall.__init__.__globals__['_owned_native']
    assert owned.__name__=='_owned_native'
    assert math._MathCall.__init__.__code__.co_filename==str(m.DEPLOYMENT/'TileOPs/benchmarks/ascend_math_more.py')
    assert b.P==Path(d)
    assert not any(n in sys.modules for n in ('torch','torch_npu','acl','ctypes'))
ok('real private loader wiring resists ambient namesake; inert ctypes token, no native build/import')
result=dict(status='passed',checks=checks,check_groups=len(checks),NPU_runs=0,native_imports=0,numeric_retests=0,accepted_added=0)
(P/'CPU_CHECK.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
