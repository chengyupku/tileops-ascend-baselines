"""Eight original DeltaNet contracts; private architecture-specific lazy registration."""
import ast, builtins, copy, hashlib, json, os, platform, sys, types
from pathlib import Path
DATA_ROOT = Path(__file__).resolve().parent
DATA_HASHES = {'CONTRACTS.json': '79c8f002bc9b0d268ef278d1df3234688c870e627e291a08392fc09a33e58c21', 'LOCAL_PINS.json': '08efbe101d7ac4e9ed86d0beb810e14e30c3c2040293cf988a4103cedb9e499f', 'RUNTIME_PINS.json': '1ec90c301865408948f289438b6264a026aebbbed7d33041f7e6863daa395ae7'}

def plain(value):
    from collections.abc import Mapping
    if hasattr(value, 'kind') and hasattr(value, 'fields'):
        return {value.kind: plain(value.fields)}
    if isinstance(value, Mapping): return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [plain(v) for v in value]
    return value

def canonical(value):
    return json.dumps(plain(value), sort_keys=True, separators=(',', ':'), allow_nan=False)

def contract(case):
    c = case._call
    return plain(dict(op=c.signature.name, case_id=case.id, call_case_id=c.case_id,
        parameters=case.params, indices=c.indices, rawparams=c.params, tensors=c.tensors,
        inputs=list(c.signature.inputs), outputs=list(c.signature.outputs),
        specs={n: None if s is None else dict(shape=s.shape,dtype=s.dtype,values=s.values,cpu=s.cpu)
               for n,s in c.specs.items()},
        presence={n:c.present(n) for n in set(c.specs)|set(c.signature.inputs)|{'out'}},has_out=c.has_out))

def key(case): return (case._call.signature.name, case.id)

def _proc(pid):
    s=Path('/proc/%d/stat'%pid).read_text().rsplit(')',1)[1].split()
    return int(s[19]),int(s[2])

def require_guard():
    import fcntl
    try:
        with Path('/run/lock/ascend-npu/device-0.lock').open('r') as f:
            m=json.load(f);child=m['child'];owner=os.environ.get('GDN_CASE_OWNER')
            if not (owner and m['owner']==owner and m['phase']=='running' and m['device']==0
                    and m['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
                    and child['pid']==os.getpgrp()
                    and _proc(child['pid'])==(child['start_ticks'],child['pid'])
                    and _proc(m['pid'])[0]==m['start_ticks']):
                raise RuntimeError('Live foreground guard identity mismatch')
            try:fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:pass
            else:
                fcntl.flock(f.fileno(),fcntl.LOCK_UN)
                raise RuntimeError('Guard kernel lock is not held')
        return m['boot_id'],m['pid'],m['start_ticks'],child['pid'],child['start_ticks'],owner
    except (OSError,AssertionError,KeyError,ValueError,TypeError) as e:
        raise RuntimeError('Live foreground npu-guard required') from e

def validate_inputs(case,args,kwargs):
    from benchmarks.ascend_baselines import UnsupportedCase
    names=list(case._call.signature.inputs)
    if kwargs or len(args)!=len(names):raise UnsupportedCase('Exact positional input arity required')
    for name,value in zip(names,args):
        spec=case._call.specs.get(name)
        if spec is None:
            if value is not None:raise UnsupportedCase('Absent optional input: '+name)
            continue
        if tuple(getattr(value,'shape',()))!=tuple(spec.shape) or str(getattr(value,'dtype','')).removeprefix('torch.')!=spec.dtype:
            raise UnsupportedCase('Exact input shape/dtype required: '+name)
        device=str(getattr(value,'device',''))
        if device!=('cpu' if spec.cpu else 'npu:0'):raise UnsupportedCase('Exact input device required: '+name)

def checked(path,digest):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=digest:raise RuntimeError('Frozen dependency changed: '+str(path))
    return raw

def metadata():return {n:json.loads(checked(DATA_ROOT/n,h)) for n,h in DATA_HASHES.items()}
def dataset(name):return json.loads(checked(DATA_ROOT/name,DATA_HASHES[name]))
def rows():return dataset('CONTRACTS.json')
def matches(case):
    try:return any(key(case)==(r['op'],r['case_id']) and canonical(contract(case))==canonical(r) for r in rows())
    except (AttributeError,KeyError,TypeError,ValueError):return False

def snapshot():
    return {n:checked(DATA_ROOT/n,h) for n,h in dataset('LOCAL_PINS.json').items()}

def implementation(run,data):
    import dataclasses
    from typing import Callable,Optional
    node=next(n for n in ast.parse(data['closure/api.py']).body if isinstance(n,ast.ClassDef) and n.name=='Implementation')
    ns=dict(dataclasses=dataclasses,Callable=Callable,Optional=Optional)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'closure/api.py','exec'),ns)
    return ns['Implementation'](run)

CACHE_NAMES=('TMPDIR','XDG_CACHE_HOME','XDG_CONFIG_HOME','ASCEND_PROCESS_LOG_PATH','ASCEND_WORK_PATH',
 'ASCEND_CACHE_PATH','TORCH_EXTENSIONS_DIR','TRITON_CACHE_DIR','TMP','TEMP','ACL_OP_COMPILER_CACHE_DIR','CCACHE_DIR')

def environment(pin):
    if platform.machine()!=pin['host']['architecture'] or platform.node()!=pin['host']['hostname']:
        raise RuntimeError('Only the exact verified architecture/host is admitted')
    if sys.prefix!=pin['python_prefix']:raise RuntimeError('Verified Python runtime required')
    for k,v in pin['environment'].items():
        if os.environ.get(k)!=v:raise RuntimeError('Verified environment required: '+k)
    if os.environ.get('LD_PRELOAD') or os.environ.get('PYTHONHOME'):raise RuntimeError('Loader override rejected')
    for k in ('ASCEND_VISIBLE_DEVICES','ASCEND_RT_VISIBLE_DEVICES'):
        if os.environ.get(k) not in (None,'','0'):raise RuntimeError('Physical NPU 0 required')
    root=Path(os.environ.get('LINEAR8_RUN_ROOT','/invalid')).resolve()
    if root.parent!=DATA_ROOT/'runs' or not (root/'RUN_EXCLUSIVE').is_file():raise RuntimeError('Private run directory required')
    for k in CACHE_NAMES:
        if Path(os.environ.get(k,'/invalid')).resolve()!=root/k:raise RuntimeError('Private cache required: '+k)
    return tuple(sorted(os.environ.items()))

def loaded_libraries(pin):
    expected={str(Path(p).resolve()) for p in pin['runtime'] if '.so' in p}
    names={Path(p).name for p in expected}
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields=line.split(maxsplit=5)
        if len(fields)!=6:continue
        path=fields[5];base=Path(path.removesuffix(' (deleted)')).name
        relevant=base in names or '/Ascend/' in path or '/torch_npu/' in path or base=='libcust_opapi.so'
        if not relevant or '.so' not in path:continue
        if path.endswith(' (deleted)') or str(Path(path).resolve()) not in expected:raise RuntimeError('Unexpected mapped native dependency: '+path)
        if Path(path).stat().st_ino!=int(fields[4]):raise RuntimeError('Mapped file replaced: '+path)

def admit(pin,full):
    guard=require_guard();env=environment(pin)
    if full:
        for path,h in pin['runtime'].items():checked(path,h)
        recipe=pin['recipe'];aliases=[str(Path(recipe[k]).resolve()) for k in ('opapi','nnopbase')]
        expected={str(Path(p).resolve()) for p in pin['runtime']}
        if not set(aliases)<=expected:raise RuntimeError('Unpinned ACL runtime recipe')
    loaded_libraries(pin)
    if require_guard()!=guard or environment(pin)!=env:raise RuntimeError('Admission changed')
    return guard,env

def private_load(pin,data):
    """Execute exact collected binding bytes, isolating direct_acl and passive API imports."""
    package=pin['package'];mods={};original=builtins.__import__
    from benchmarks.ascend_baselines import Candidate,UnsupportedCase
    public=types.SimpleNamespace(Candidate=Candidate,UnsupportedCase=UnsupportedCase)
    api=types.SimpleNamespace(Implementation=type(implementation(lambda:None,data)))
    def load(name):
        if name in mods:return mods[name]
        path=package+'/'+name+'.py';raw=data[path]
        m=types.ModuleType('_linear8_'+name);m.__file__=str(DATA_ROOT/path)
        m.__dict__['__builtins__']=dict(vars(builtins),__import__=private_import);mods[name]=m
        exec(compile(raw,m.__file__,'exec'),m.__dict__)
        if name=='direct_acl':m.require_guard=require_guard
        return m
    def private_import(name,globals=None,locals=None,fromlist=(),level=0):
        if name=='direct_acl':return load(name)
        if name=='benchmarks.api':return api
        if name=='benchmarks.ascend_baselines':return public
        return original(name,globals,locals,fromlist,level)
    return load('binding')

def private_build(case,pin,data):
    require_guard()
    return private_load(pin,data).build(case)

def make_candidate(case):
    from benchmarks.ascend_baselines import Candidate,UnsupportedCase
    if not matches(case):raise UnsupportedCase('Exact DeltaNet manifest contract required')
    frozen=copy.deepcopy(case);expected=canonical(contract(case));data=snapshot()
    pin=dataset('RUNTIME_PINS.json')[case.id]
    state={'inner':None,'closed':False,'guard':None,'env':None}
    def identity():
        if require_guard()!=state['guard'] or environment(pin)!=state['env']:raise RuntimeError('Original guard/environment lifetime required')
    def cleanup():
        inner=state['inner']
        if inner is not None:identity()
        state['inner']=None;state['closed']=True
        if inner is not None and inner.cleanup is not None:inner.cleanup()
    def run(*args,**kwargs):
        if state['closed']:raise RuntimeError('Candidate is closed')
        if canonical(contract(case))!=expected:raise UnsupportedCase('Case mutated after construction')
        try:
            validate_inputs(frozen,args,kwargs)
            if state['inner'] is not None:identity()
            guard,env=admit(pin,state['inner'] is None)
            if state['inner'] is None:
                state['guard']=guard;state['env']=env
                # Original binding reads its small recipe/contract files at first build.
                for name,raw in data.items():
                    if name.endswith('.json'):checked(DATA_ROOT/name,hashlib.sha256(raw).hexdigest())
                state['inner']=private_build(frozen,pin,data)
            identity();loaded_libraries(pin)
            # Equal metadata copies are a contract requirement, never a numerical oracle.
            if frozen.id=='varlen-4x1k-bfloat16':
                bounds=args[5].cpu().tolist();host=args[6].tolist()
                if bounds!=[0,1024,2048,3072,4096] or host!=bounds:raise UnsupportedCase('Exact equal packed boundaries required')
            result=state['inner'].implementation.run(*args,**kwargs)
            identity();loaded_libraries(pin)
            return result
        except BaseException as error:
            try:cleanup()
            except BaseException as clean_error:raise clean_error from error
            raise
    api='aclnnRecurrentKda' if case.id.startswith('decode-') else 'aclnnChunkGatedDeltaRule'
    return Candidate(implementation(run,data),'ops-transformer::'+api,
        'Exact original full API; architecture-specific frozen binding; guarded lazy lifetime.',
        ({str(DATA_ROOT/n):hashlib.sha256(raw).hexdigest() for n,raw in data.items()} | pin['runtime']),timing_mode='host_roundtrip',cleanup=cleanup)

make=make_candidate

def wrap(previous):
    def factory(case):
        if key(case) in {(r['op'],r['case_id']) for r in rows()}:return make_candidate(case)
        if previous is None:
            from benchmarks.ascend_baselines import UnsupportedCase
            raise UnsupportedCase('No previous provider for unknown key')
        return previous(case)
    factory.previous=previous
    return factory

def install(providers):
    providers['DeltaNetFwdOp']=wrap(providers.get('DeltaNetFwdOp'))
