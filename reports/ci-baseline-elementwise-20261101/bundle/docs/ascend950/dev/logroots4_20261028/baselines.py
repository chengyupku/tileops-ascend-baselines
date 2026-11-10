"""Same-host original baseline timing; rejects Event and records failures."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(ROOT),str(ROOT/'TileOPs'),str(ROOT/'TileOPs/src'),str(HERE.with_name('arithmetic9_20261027'))]
p=argparse.ArgumentParser();p.add_argument('--op',default='AddFwdOp');p.add_argument('--library',required=True);p.add_argument('--regex',default='');p.add_argument('--output',required=True);p.add_argument('--variant',default='simt');p.add_argument('--exported-flush',action='store_true');p.add_argument('--no-time',action='store_true')
a=p.parse_args()
RUNNER_HASH=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
if a.library=='tilelang-ascend':
    from benchmarks.ascend_all_pairs.tla import native
    native.prepare_deployment()
import torch
import torch_npu
def forbidden(*a,**kw):raise AssertionError('Event forbidden')
torch.npu.Event=forbidden
from benchmarks.api import cases
from workloads.device import set_run_device
from tools import measure_baselines as timing
from tools.baseline_protocol import seed_case
from full_digest import tensor_digest, PROTOCOL as DIGEST_PROTOCOL
timing._tensor_digest=tensor_digest
if a.library in {'tilelang-ascend', 'ops-math', 'ops-nn'} or a.exported_flush:
    data=json.loads((HERE.with_name('arithmetic9_20261027')/'flush/export.json').read_text());lib=ctypes.CDLL(data['library'])
    assert hashlib.sha256(Path(data['library']).read_bytes()).hexdigest()==data['sha256']
    lib.get_last_error.restype=ctypes.c_char_p;assert lib.init()==0
    lib.call.argtypes=[ctypes.c_void_p,ctypes.c_void_p]
    def flush(buffer):
        rc=lib.call(buffer.data_ptr(),torch.npu.current_stream().npu_stream)
        if rc:raise RuntimeError(lib.get_last_error().decode())
    timing._flush_kernel_cache[torch.npu.current_device()]=(flush,torch.empty(268435456,device='npu',dtype=torch.int32))
torch.npu.set_device(0);set_run_device('npu')
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
import platform,time
properties=torch.npu.get_device_properties(0)
environment={'host':platform.node(),'machine':platform.machine(),'torch':torch.__version__,'torch_npu':torch_npu.__version__,'device_name':torch.npu.get_device_name(0),'device_properties':str(properties),'environment':{k:os.environ.get(k) for k in ['ASCEND_HOME_PATH','ASCEND_TOOLKIT_HOME','BISHENG_HOME','ASCEND_CUSTOM_OPP_PATH']},'timer_sha256':hashlib.sha256(Path(timing.__file__).read_bytes()).hexdigest(),'runner_sha256':RUNNER_HASH}
for path in [Path('/etc/ascend-npu-guard-abi.json'),Path('/usr/local/Ascend/driver/version.info'),Path('/usr/local/Ascend/cann-9.1.0/version.cfg')]:
    if path.is_file():environment[str(path)]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'text':path.read_text()}
env_path=HERE/'environment'/(a.op+'-'+a.library+'-'+str(time.time_ns())+'.json');env_path.parent.mkdir(exist_ok=True);env_path.write_text(json.dumps(environment,indent=2))

for case in cases(a.op):
    if a.regex and not __import__('re').search(a.regex,case.id):continue
    row=dict(op=a.op,case_id=case.id,library=a.library,params=case.params,environment_receipt=str(env_path),measurement_requested=not a.no_time);candidate=None
    try:
        seed_case(torch,a.op,case.id);xs=case.inputs;before=timing._tensor_digest(xs)
        ref=case.reference(*(x.cpu() for x in xs))
        if a.library=='tilelang-ascend':
            from benchmarks.ascend_all_pairs.tla.adapters import elementwise
            candidate=elementwise(a.op,case)
        elif a.library=='ops-math':
            if a.op=='AddFwdOp':
                from benchmarks.ascend_all_pairs.ops_math.specs import SPECS
                SPECS['AddFwdOp']={'directory':'math/add','graph':'Add'}
            from benchmarks.ascend_all_pairs.ops_math.graph import make
            candidate=make(case,a.op)
        elif a.library=='ops-nn':
            if a.op == 'RsqrtFwdOp':
                from original_nn_adapter.reductions import rsqrt
                candidate=rsqrt(case)
            else:
                from original_nn_adapter.elementwise import foreach
                candidate=foreach(case)
        elif a.library=='tilelang' and a.op=='AddFwdOp':
            import importlib.util
            import tilelang
            from benchmarks.api import Implementation
            from benchmarks.ascend_baselines import Candidate
            src=ROOT/'tilelang/examples/ascend/example_vecadd.py'
            spec=importlib.util.spec_from_file_location('arithmetic9_original_vecadd',src)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            shape=tuple(ref.shape);size=ref.numel();padded=(size+8192*64-1)//(8192*64)*(8192*64)
            kernel=tilelang.compile(module.vector_add(padded,mode=a.variant),target='ascend',out_idx=-1)
            def run(x,y):
                packed=[torch.nn.functional.pad(t.expand(shape).contiguous().reshape(-1).float(),(0,padded-size)) for t in (x,y)]
                return kernel(*packed)[:size].to(x.dtype).reshape(shape)
            candidate=Candidate(Implementation(run),'original example_vecadd.vector_add('+a.variant+')',
                'Per-call exact FP32 widening, broadcast materialization, independent zero padding and output crop/cast.',
                {str(src):hashlib.sha256(src.read_bytes()).hexdigest()})
            row['variant']=a.variant
        else:
            from benchmarks.ascend_all_pairs.registry import candidate_factory
            candidate=candidate_factory(case,a.library)()
        assert candidate.timing_mode=='device_async','Child-process wall-clock baseline cannot use this device timer'
        fn=candidate.implementation.run;got=fn(*xs);torch.npu.synchronize()
        exact_contract=a.op in {'FloorDivideFwdOp','RemainderFwdOp','MaximumFwdOp','MinimumFwdOp'} or (a.op=='DivFwdOp' and case.params.get('rounding_mode') is not None)
        tol=0.0 if exact_contract else {torch.float16:1e-3,torch.bfloat16:1.6e-2,torch.float32:1e-5}[ref.dtype]
        torch.testing.assert_close(got.cpu(),ref,rtol=tol,atol=tol,equal_nan=True)
        row.update(correct=True,correctness_protocol='manifest-exact-v1' if exact_contract else 'manifest-dtype-tolerance-v1',correctness_tolerance={'rtol':tol,'atol':tol},api=candidate.api,adaptation=candidate.adaptation,sources=candidate.source_sha256)
        if not a.no_time:
            row.update(timing._time_reference(torch,torch_npu,fn,tuple(xs),3,15))
        else:
            for _ in range(20):fn(*xs)
            torch.npu.synchronize()
        assert timing._tensor_digest(xs)==before,'input mutation'
        row.update(status='correctness_only' if a.no_time else 'measured',input_unchanged=True,input_digest_protocol=DIGEST_PROTOCOL,input_digest=before)
        row['loaded_custom_libraries'] = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if '/vendors/' in line and '.so' in line})
    except Exception:row.update(status='failed',error=traceback.format_exc())
    finally:
        if candidate and candidate.cleanup:candidate.cleanup()
        row['torch']=torch.__version__;row['adapter_sha256']=RUNNER_HASH
        with out.open('a') as f:f.write(json.dumps(row,default=str)+'\n');f.flush()
        print(json.dumps(row,default=str),flush=True)
torch.npu.synchronize()
