"""Three exact TileLang 0.1.15 Ascend contracts; original factories remain runtime-lazy."""
import ast
import builtins
import copy
import hashlib
import json
import os
import platform
import sys
import tempfile
import types
from pathlib import Path

LOCAL_ROOT = Path('/root/dyq/workspace/reports/baseline-integration-700-20261026/threehost_tilelang3_complete_capture')
REMOTE_ROOT = Path('/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_tilelang3_complete_capture')
# Explicit supported deployment roots, independent of the published module location.
DATA_ROOT = REMOTE_ROOT if platform.machine() == 'aarch64' else LOCAL_ROOT
WORKSPACE = '/opt/tileops-all-pairs/workspace'
REPORT = WORKSPACE + '/reports/baseline-integration-700-20261026'
BASE = WORKSPACE + '/TileOPs/benchmarks/ascend_baselines.py'
API = WORKSPACE + '/TileOPs/benchmarks/api.py'
DATA_HASHES = {'CONTRACTS.json': '8ea7db91f80f22729541ec7a49bf1fb9ded7ef9f92f26314a3942bcb1f0e928a', 'PINS.json': '0531a288fac68510f048854c8c0aa099eba0631cdedaaf949cee65adc7a51df5'}


def checked(path, digest):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise RuntimeError('Frozen dependency changed: ' + str(path))
    return raw


def metadata():
    return {n: json.loads(checked(DATA_ROOT/n, h)) for n, h in DATA_HASHES.items()}


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


def rows(): return metadata()['CONTRACTS.json']


def matches(case):
    try:
        return any(key(case)==(r['op'],r['case_id']) and canonical(contract(case))==canonical(r) for r in rows())
    except (AttributeError, KeyError, TypeError, ValueError): return False


def snapshot(runtime=False):
    pins=metadata()['PINS.json'];data={}
    for original, item in pins['sources'].items():
        data[original]=checked(DATA_ROOT/item['file'],item['sha256'])
    if runtime:
        for path,digest in pins['runtime'].items(): checked(path,digest)
    return data


def _implementation(run, data):
    # Exact passive original API class; importing full benchmarks.api imports torch.
    from typing import Callable, Optional
    import dataclasses
    tree=ast.parse(data[API],API)
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Implementation')
    ns=dict(dataclasses=dataclasses,Callable=Callable,Optional=Optional)
    exec(compile(ast.Module(body=[node],type_ignores=[]),API,'exec'),ns)
    return ns['Implementation'](run)


def private_load(data):
    import importlib, importlib.util
    original_import=builtins.__import__
    def spec(name,path):
        path=str(path)
        if path not in data:raise RuntimeError('Unpinned source import: '+path)
        class Loader:
            def create_module(self,spec):return None
            def exec_module(self,mod):
                mod.__file__=path
                exec(compile(data[path],path,'exec'),mod.__dict__)
        return importlib.util.spec_from_loader(name,Loader())
    bridge=types.SimpleNamespace(util=types.SimpleNamespace(spec_from_file_location=spec,module_from_spec=importlib.util.module_from_spec))
    def private_import(name,globals=None,locals=None,fromlist=(),level=0):
        if name in ('importlib','importlib.util'):return bridge
        return original_import(name,globals,locals,fromlist,level)
    def load(path):
        mod=types.ModuleType('_tilelang3_original');mod.__file__=path
        mod.__dict__['__builtins__']=dict(vars(builtins),__import__=private_import)
        exec(compile(data[path],path,'exec'),mod.__dict__)
        return mod
    return load


def _proc(pid):
    s=Path('/proc/%d/stat'%pid).read_text().rsplit(')',1)[1].split()
    return int(s[19]),int(s[2])


def require_guard():
    import fcntl
    try:
        with Path('/run/lock/ascend-npu/device-0.lock').open('r') as f:
            m=json.load(f);child=m['child'];owner=os.environ.get('TILELANG3_GUARD_OWNER')
            assert owner and m['owner']==owner and m['phase']=='running' and m['device']==0
            assert m['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            assert child['pid']==os.getpgrp()
            assert _proc(child['pid'])==(child['start_ticks'],child['pid'])
            assert _proc(m['pid'])[0]==m['start_ticks']
            try:fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:pass
            else:
                fcntl.flock(f.fileno(),fcntl.LOCK_UN)
                raise RuntimeError('Guard kernel lock is not held')
        return m['boot_id'],m['pid'],m['start_ticks'],child['pid'],child['start_ticks'],owner
    except (OSError,AssertionError,KeyError,ValueError,TypeError) as e:
        raise RuntimeError('Live foreground npu-guard required') from e


CACHE_NAMES=('ASCEND_CACHE_PATH','ASCEND_WORK_PATH','XDG_CACHE_HOME','TORCH_EXTENSIONS_DIR',
 'ASCEND_PROCESS_LOG_PATH','TRITON_CACHE_DIR','TMPDIR','TMP','TEMP','ACL_OP_COMPILER_CACHE_DIR','CCACHE_DIR','TILELANG_CACHE_DIR')


def environment():
    pins=metadata()['PINS.json']
    if platform.machine()!=pins['host']['machine'] or platform.node()!=pins['host']['node']:
        raise RuntimeError('Only tested server1 ARM host is supported')
    if DATA_ROOT!=REMOTE_ROOT or sys.prefix!='/opt/tileops-ci/runtime':
        raise RuntimeError('Tested deployment/runtime required')
    for name,value in pins['environment'].items():
        if os.environ.get(name)!=value:raise RuntimeError('Tested environment required: '+name)
    for name in ('LD_PRELOAD','PYTHONHOME','TILELANG_LIBRARY_PATH','TVM_LIBRARY_PATH','TVM_HOME','TVM_IMPORT_PYTHON_PATH','TL_ROOT','TL_PLATFORM','ASCEND_NPU_ARCH'):
        if os.environ.get(name):raise RuntimeError('Unreviewed loader override: '+name)
    for name in ('ASCEND_RT_VISIBLE_DEVICES','ASCEND_VISIBLE_DEVICES'):
        if os.environ.get(name) not in (None,'','0'):raise RuntimeError('Physical device 0 required')
    root=Path(os.environ.get('TILELANG3_RUN_ROOT','/invalid')).resolve()
    if root.parent!=REMOTE_ROOT/'runs' or not (root/'RUN_EXCLUSIVE').is_file():
        raise RuntimeError('Fresh private recipe run directory required')
    for name in CACHE_NAMES:
        if Path(os.environ.get(name,'/invalid')).resolve()!=root/name:
            raise RuntimeError('Private cache/log directory required: '+name)
    return tuple(sorted(os.environ.items())),str(root)


def admit(runtime=True):
    environment_token=environment() # wrong host fails before imports or native access
    guard=require_guard()
    if not metadata()['PINS.json']['publication_ready']:
        raise RuntimeError('Host compiler pre-freeze procedure deviation remains for publication review; execution disabled')
    data=snapshot(runtime=runtime)
    loaded_libraries()
    if require_guard()!=guard or environment()!=environment_token:raise RuntimeError('Admission changed')
    return data,guard,environment_token


def validate_inputs(case,args,kwargs):
    from benchmarks.ascend_baselines import UnsupportedCase
    names=[n for n in case._call.signature.inputs if case._call.present(n)]
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


def mapped_identity(path, inode, device):
    """Verify same file via a passive read-only mmap, including overlay backing devices."""
    import mmap
    with Path(path).open('rb') as stream:
        st=os.fstat(stream.fileno())
        if st.st_ino != inode:return False
        if (os.major(st.st_dev),os.minor(st.st_dev)) == device:return True
        before=set(Path('/proc/self/maps').read_text().splitlines())
        with mmap.mmap(stream.fileno(), min(st.st_size,4096), access=mmap.ACCESS_READ):
            after=set(Path('/proc/self/maps').read_text().splitlines())
            for line in after-before:
                fields=line.split(maxsplit=5)
                if len(fields)!=6:continue
                if (fields[5]==str(Path(path).resolve()) and int(fields[4])==inode
                    and tuple(int(x,16) for x in fields[3].split(':'))==device):return True
        return False


def loaded_libraries():
    pins=metadata()['PINS.json']['runtime']
    expected={str(Path(p).resolve()):h for p,h in pins.items() if '.so' in p}
    names={Path(p).name for p in expected}
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields=line.split(maxsplit=5)
        if len(fields)!=6:continue
        mapped=fields[5].removesuffix(' (deleted)')
        relevant=Path(mapped).name in names or ('site-packages/tilelang/' in mapped and '.so' in mapped)
        if not relevant:continue
        path=fields[5]
        if path.endswith(' (deleted)') or str(Path(path).resolve()) not in expected:
            raise RuntimeError('Unexpected mapped dependency: '+path)
        if not mapped_identity(path,int(fields[4]),tuple(int(x,16) for x in fields[3].split(':'))):raise RuntimeError('Mapped dependency replaced: '+path)


def make_candidate(case):
    from benchmarks.ascend_baselines import Candidate, UnsupportedCase
    if not matches(case):raise UnsupportedCase('Exact tilelang3 manifest contract required')
    expected=canonical(contract(case));frozen=types.SimpleNamespace(_call=copy.deepcopy(case._call),id=case.id,params=copy.deepcopy(case.params));data=snapshot()
    state={'inner':None,'closed':False,'guard':None,'environment':None}
    def cleanup():
        inner=state['inner']
        if state['guard'] is not None:
            if require_guard()!=state['guard'] or environment()!=state['environment']:
                raise RuntimeError('Cleanup requires original live guard and environment')
        state['inner']=None;state['closed']=True
        if state['guard'] is not None:
            torch=sys.modules.get('torch')
            if torch is not None and torch.npu.is_initialized():torch.npu.synchronize()
        if inner is not None and inner.cleanup is not None:inner.cleanup()
    def run(*args,**kwargs):
        if state['closed']:raise RuntimeError('Candidate is closed')
        try:
            if canonical(contract(case))!=expected or not matches(case):raise UnsupportedCase('Case changed after construction')
            environment()
            validate_inputs(frozen,args,kwargs)
            if state['inner'] is not None:
                if require_guard()!=state['guard'] or environment()!=state['environment']:
                    raise RuntimeError('Candidate cannot cross guard/environment lifetimes')
            data,guard,env=admit(runtime=state['inner'] is None)
            if state['inner'] is None:
                state['guard']=guard;state['environment']=env
                load=private_load(data)
                mod=load(ADAPTER)
                frozen.inputs=args
                out=Path(tempfile.mkdtemp(prefix='candidate-',dir=env[1]))
                state['inner']=mod.factory(frozen,key(frozen)[0],out)
            if require_guard()!=guard or environment()!=env:raise RuntimeError('Execution identity changed')
            loaded_libraries()
            result=state['inner'].implementation.run(*args,**kwargs)
            loaded_libraries()
            if require_guard()!=guard or environment()!=env:raise RuntimeError('Execution lifetime changed')
            return result
        except BaseException as error:
            try:cleanup()
            except BaseException as clean_error:raise clean_error from error
            raise
    return Candidate(_implementation(run,data),'tilelang::original-ascend',
        'Exact tilelang contract; unchanged original adapter; lazy guarded original TileLang 0.1.15 Ascend.',
        {p:hashlib.sha256(raw).hexdigest() for p,raw in data.items()},cleanup=cleanup)


make=make_candidate

def wrap(previous):
    def factory(case):
        from benchmarks.ascend_baselines import UnsupportedCase
        if key(case) in {(r['op'],r['case_id']) for r in rows()}:return make_candidate(case)
        if previous is None:raise UnsupportedCase('No previous provider for unknown key')
        return previous(case)
    factory.previous=previous
    return factory


def install(providers):
    """Parent invokes against the actual provider dictionary when publishing."""
    for op in {r['op'] for r in rows()}:providers[op]=wrap(providers.get(op))

ADAPTER = '/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_tla_other54_r2/tilelang_adapter.py'
