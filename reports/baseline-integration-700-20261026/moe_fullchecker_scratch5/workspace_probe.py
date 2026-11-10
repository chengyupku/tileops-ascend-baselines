#!/usr/bin/env python3
"""Default is CPU-only JSON plan. --device-query is ONLY for parent's later npu-guard reservation.
No torch, kernels, workspace allocations, tensor reads, or fabricated device pointers.
One case/operand per process; real full-size ND storage. Uninitialized storage is safe
for workspace planning, which must not read tensor contents. Does not prove torch dispatch.
"""
import argparse
import json
import math
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--key', default='ds-v3-prefill-mlp-up-bfloat16')
    p.add_argument('--operand', choices=['A', 'B', 'metadata', 'checker'], default='B')
    p.add_argument('--device-query', action='store_true')
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--sdk', default='/usr/local/Ascend/cann-9.2.0-beta.2')
    a = p.parse_args()
    rows = json.loads(Path(__file__).with_name('PER_KEY_BOUNDS.json').read_text())['keys']
    row = next(r for r in rows if r['case_id'] == a.key)
    spec = row['probe_descriptors'][a.operand]
    shape, dtype = spec['shape'], spec['dtype']
    n = math.prod(shape)
    code, width = {'bfloat16': (27, 2), 'float32': (0, 4), 'int32': (3, 4)}[dtype]
    plan = dict(case_id=a.key, operand=a.operand, **spec,
                allocation_bytes=2*n*width+n+1, kernel_launches=0,
                device_query_requested=a.device_query)
    print(json.dumps(plan), flush=True)
    if not a.device_query:
        return
    # This entire branch needs parent's explicit release and npu-guard run.
    import ctypes as C
    V, U, I = C.c_void_p, C.c_uint64, C.c_int64
    P = C.POINTER
    libs = Path(a.sdk) / 'lib64'
    rt = C.CDLL(str(libs / 'libascendcl.so'), mode=C.RTLD_GLOBAL)
    base = C.CDLL(str(libs / 'libnnopbase.so'), mode=C.RTLD_GLOBAL)
    api = C.CDLL(str(libs / 'libopapi.so'), mode=C.RTLD_GLOBAL)
    def bind(lib, name, args, result=C.c_int):
        f = getattr(lib, name)
        f.argtypes, f.restype = args, result
        return f
    init = bind(rt, 'aclInit', [C.c_char_p])
    device = bind(rt, 'aclrtSetDevice', [C.c_int])
    malloc = bind(rt, 'aclrtMalloc', [P(V), C.c_size_t, C.c_int])
    free = bind(rt, 'aclrtFree', [V])
    sync = bind(rt, 'aclrtSynchronizeDevice', [])
    finalize = bind(rt, 'aclFinalize', [])
    create = bind(base, 'aclCreateTensor', [P(I), U, C.c_int, P(I), I, C.c_int, P(I), U, V], V)
    destroy = bind(base, 'aclDestroyTensor', [V])
    array = bind(base, 'aclCreateIntArray', [P(I), U], V)
    destroy_array = bind(base, 'aclDestroyIntArray', [V])
    destroy_executor = bind(base, 'aclDestroyAclOpExecutor', [V])
    close = bind(api, 'aclnnIsCloseGetWorkspaceSize', [V,V,C.c_double,C.c_double,C.c_bool,V,P(U),P(V)])
    all_ = bind(api, 'aclnnAllGetWorkspaceSize', [V,V,C.c_bool,V,P(U),P(V)])
    def check(rc, label):
        if rc: raise RuntimeError(f'{label}: {rc}')
    buffers, tensors, executors, arrays, keep = [], [], [], [], []
    initialized = active = False
    try:
        check(init(None), 'aclInit'); initialized = True
        check(device(a.device), 'aclrtSetDevice'); active = True
        def tensor(dims, dtype_code, itemsize):
            storage = V()
            check(malloc(C.byref(storage), max(1, math.prod(dims)*itemsize), 2), 'aclrtMalloc')
            if not storage.value: raise RuntimeError('allocator returned NULL')
            buffers.append(storage)
            strides = [math.prod(dims[i+1:]) for i in range(len(dims))]
            ds, st = (I*len(dims))(*dims), (I*len(dims))(*strides)
            keep.extend([ds, st])
            t = create(ds, len(dims), dtype_code, st, 0, 2, ds, len(dims), storage)
            if not t: raise RuntimeError('aclCreateTensor returned NULL')
            tensors.append(t)
            return t
        x, y = tensor(shape, code, width), tensor(shape, code, width)
        matches, scalar = tensor(shape, 12, 1), tensor([], 12, 1)
        dims_storage = (I*len(shape))(*range(len(shape))); keep.append(dims_storage)
        dims = array(dims_storage, len(shape))
        if not dims: raise RuntimeError('aclCreateIntArray returned NULL')
        arrays.append(dims)
        for name, f, args in [('aclnnIsClose', close, [x,y,spec['rtol'],spec['atol'],spec['equal_nan'],matches]),
                               ('aclnnAll', all_, [matches,dims,False,scalar])]:
            size, ex = U(), V()
            rc = f(*args, C.byref(size), C.byref(ex))
            if ex.value: executors.append(ex)
            print(json.dumps(dict(api=name, status=rc, workspace_bytes=size.value,
                                  executor_nonnull=bool(ex.value))), flush=True)
            check(rc, name+'GetWorkspaceSize')
    finally:
        if active: check(sync(), 'aclrtSynchronizeDevice')
        for ex in reversed(executors): check(destroy_executor(ex), 'aclDestroyAclOpExecutor')
        for t in reversed(tensors): check(destroy(t), 'aclDestroyTensor')
        for d in reversed(arrays): check(destroy_array(d), 'aclDestroyIntArray')
        for b in reversed(buffers): check(free(b), 'aclrtFree')
        # Do not reset the shared device. Process-local finalize only.
        if initialized: check(finalize(), 'aclFinalize')

if __name__ == '__main__':
    main()
