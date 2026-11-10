import ast,copy,hashlib,json,os,sys,types,platform
from pathlib import Path
DATA_ROOT=Path(__file__).resolve().parent
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
            m=json.load(f);child=m['child'];owner=os.environ.get('PTO23_OWNER')
            if not (owner=='baseline700-20261026/threehost_pto_registration23' and m['owner']==owner and m['phase']=='running' and m['device']==0
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

def implementation(run,data):
    import dataclasses
    from typing import Callable,Optional
    node=next(n for n in ast.parse(data['closure/api.py']).body if isinstance(n,ast.ClassDef) and n.name=='Implementation')
    ns=dict(dataclasses=dataclasses,Callable=Callable,Optional=Optional)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'closure/api.py','exec'),ns)
    return ns['Implementation'](run)

def dataset(name):
    pins=json.loads(checked(DATA_ROOT/'DATA_HASHES.json','6cba138e106d1c2bda542193f9b801efbaccdb39cfd36e18d7c3f6d9edfb1863'))
    return json.loads(checked(DATA_ROOT/name,pins[name]))

def rows():return dataset('CONTRACTS.json')
def matches(case):
    try:return any(canonical(contract(case))==canonical(r) for r in rows())
    except (AttributeError,KeyError,ValueError,TypeError):return False

def load(name,path):
    import importlib.util
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return mod

def current_host():
    for name,p in dataset('HOSTS.json').items():
        if p['hostname']==platform.node():return name,p
    raise RuntimeError('Unadmitted host')

def mapped(pin):
    expected={str(Path(p).resolve()):h for p,h in pin['pins'].items() if Path(p).name in ('libtilelang_module.so','libtvm.so','libtvm_runtime.so')}
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields=line.split(maxsplit=5)
        if len(fields)!=6:continue
        path=fields[5]
        if Path(path.removesuffix(' (deleted)')).name not in ('libtilelang_module.so','libtvm.so','libtvm_runtime.so'):continue
        if path.endswith(' (deleted)') or str(Path(path).resolve()) not in expected:raise RuntimeError('Wrong loaded compiler')
        if Path(path).stat().st_ino!=int(fields[4]):raise RuntimeError('Replaced mapped compiler')
        checked(path,expected[str(Path(path).resolve())])

def build(case,out):
    require_guard();host,pin=current_host()
    if sys.version!=pin['python']:raise RuntimeError('Verified Python ABI/runtime required')
    if os.environ.get('LD_PRELOAD') or os.environ.get('PYTHONHOME'):raise RuntimeError('Loader override forbidden')
    for p,h in pin['pins'].items():checked(p,h)
    for p,h in dataset('LOCAL_PINS.json').items():checked(DATA_ROOT/p,h)
    P=Path(pin['base']);W=Path(pin['workspace']);V=Path(pin['vendor']);C=Path(pin['compiler'])
    setup=load('_pto23_setup',DATA_ROOT/('setup_'+host+'.py'))
    setup.setup(P,W,V,C,P/'runtime',out)
    import setuptools
    import tilelang
    if Path(tilelang.__file__).resolve()!=(P/'runtime/tilelang/__init__.py').resolve():raise RuntimeError('Wrong TileLang frontend')
    mapped(pin)
    pkg=types.ModuleType('_pto23_original');pkg.__path__=[str(DATA_ROOT/'closure/private_tla')];sys.modules[pkg.__name__]=pkg
    load(pkg.__name__+'.deployment',DATA_ROOT/'closure/private_tla/deployment.py')
    native=load(pkg.__name__+'.native',DATA_ROOT/'closure/private_tla/native.py')
    native.VENDOR=P/'original';native.runtime=lambda:tilelang
    adapter=load('_pto23_adapter',DATA_ROOT/'closure/adapter.py')
    inner=adapter.factory(case,case._call.signature.name,native,tilelang,out)
    mapped(pin)
    return inner,pin

def make(case):
    from benchmarks.ascend_baselines import Candidate,UnsupportedCase
    if not matches(case):raise UnsupportedCase('Exact original PTO manifest required')
    expected=canonical(contract(case));frozen=copy.deepcopy(case)
    data={n:checked(DATA_ROOT/n,h) for n,h in dataset('LOCAL_PINS.json').items()}
    state={'inner':None,'closed':False,'guard':None,'env':None,'cwd':None,'pin':None}
    def identity():
        if require_guard()!=state['guard'] or tuple(sorted(os.environ.items()))!=state['env'] or os.getcwd()!=state['cwd']:
            raise RuntimeError('Original guard/environment/cwd lifetime required')
    def cleanup():
        if state['closed']:return
        if state['inner'] is not None:
            identity()
            import torch
            torch.npu.synchronize()
            inner=state['inner'];state['inner']=None;state['closed']=True
            if inner.cleanup is not None:inner.cleanup()
        state['closed']=True
    def run(*args,**kwargs):
        if state['closed']:raise RuntimeError('Candidate closed')
        if canonical(contract(case))!=expected:raise UnsupportedCase('Mutated contract')
        validate_inputs(frozen,args,kwargs)
        require_guard()
        if state['inner'] is None:
            import tempfile
            # Keep sockets/caches short; directory is owned by this guarded process.
            root=Path(os.environ['PTO23_RUN_ROOT']).resolve()
            if root.parent!=DATA_ROOT/'runs' or not root.is_dir():raise RuntimeError('Private launch root required')
            out=Path(tempfile.mkdtemp(prefix='k',dir=root));os.chdir(out)
            state['guard']=require_guard()
            try:state['inner'],state['pin']=build(case,out)
            except BaseException:
                state['closed']=True
                torch=sys.modules.get('torch')
                if torch is not None and torch.npu.is_initialized():torch.npu.synchronize()
                raise
            state['env']=tuple(sorted(os.environ.items()));state['cwd']=os.getcwd()
        identity();mapped(state['pin'])
        try:return state['inner'].implementation.run(*args,**kwargs)
        except BaseException:
            cleanup();raise
    return Candidate(implementation(run,data),'tilelang-ascend::'+('examples/batch_gemm/batch_gemm.py:batch_matmul' if key(case)[0]=='BmmFwdOp' else 'examples/gemm/example_gemm.py:matmul'),'Original byte-identical adapter, public PTO/A5; lazy guarded compiler binding.',{str(DATA_ROOT/n):hashlib.sha256(v).hexdigest() for n,v in data.items()},cleanup=cleanup)

def wrap(previous):
    def factory(case):
        if key(case) in {(r['op'],r['case_id']) for r in rows()}:return make(case)
        if previous is not None:return previous(case)
        from benchmarks.ascend_baselines import UnsupportedCase
        raise UnsupportedCase('Unknown PTO exact key')
    return factory

def install(providers):
    for op in ('BmmFwdOp','GemmFwdOp'):providers[op]=wrap(providers.get(op))
