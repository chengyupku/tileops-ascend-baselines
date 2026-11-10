"""One original V5 workspace-size query. No workspace allocation or kernel launch."""
import ctypes as C
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback

P = Path(__file__).resolve().parent
W = Path('/root/dyq/workspace')
KEEP = []

def digest(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def main():
    name, out_arg = sys.argv[1:]
    out = Path(out_arg)
    out.mkdir(parents=True, exist_ok=True)
    row = next(x for x in json.loads((P/'CASES.json').read_text()) if x['name'] == name)
    release = json.loads((P/'PARENT_RELEASE.json').read_text())
    guard = json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text())
    assert guard['phase'] == 'running' and guard['owner'] == release['owner']
    assert guard['child']['pid'] == os.getpgrp()
    assert guard['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    assert all(digest(p) == h for p, h in json.loads((P/'PINS.json').read_text()).items())
    report = dict(shape=row, guard=guard, kernel_launches=0, query_calls=0,
                  workspace_allocations=0, started=time.time())
    def save():
        (out/'RESULT.json').write_text(json.dumps(report, indent=2)+'\n')
    save()
    try:
        gate_path = P.parent/'bf16_moe_masked_recovery1/resource_gate.py'
        spec = importlib.util.spec_from_file_location('diagnostic_resource_gate', gate_path)
        gate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gate)
        report['host'] = gate.capacity()
        assert report['host']['effective_available_host_bytes'] >= release['host_free_policy_bytes']
        import torch
        import torch_npu
        sys.path[:0] = [str(W/'TileOPs'), str(W/'TileOPs/src')]
        from benchmarks.ascend_math_more import _MathCall
        torch.npu.synchronize()
        torch.npu.reset_peak_memory_stats()
        def sample():
            free, total = torch.npu.mem_get_info()
            return dict(free=int(free), total=int(total), allocated=torch.npu.memory_allocated(),
                        reserved=torch.npu.memory_reserved(), peak_allocated=torch.npu.max_memory_allocated(),
                        peak_reserved=torch.npu.max_memory_reserved(), host_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        report['before_buffers'] = sample()
        assert report['before_buffers']['free'] >= release['device_free_policy_bytes']
        m, k, n = (row[x] for x in ('M', 'K', 'N'))
        # This is the exact packed native boundary, after original b[e].T.contiguous().float().
        x = torch.empty((m,k), dtype=torch.float32, device='npu')
        weight = torch.empty((k,n), dtype=torch.float32, device='npu')
        y = torch.empty((m,n), dtype=torch.float32, device='npu')
        assert sum(t.numel()*t.element_size() for t in (x,weight,y)) == row['payload_bytes']
        host = P.parent/'bf16_local_gemm18/install/vendors/bf16gmm_transformer/op_api/lib/libcust_opapi.so'
        bridge = _MathCall('aclnnGroupedMatmulV5', [C.c_void_p]*12+[C.c_int64]*4+[C.c_void_p]*4,
                           library='ops-transformer', host_library=host)
        bridge.base.aclCreateTensorList.argtypes = [C.POINTER(C.c_void_p), C.c_uint64]
        bridge.base.aclCreateTensorList.restype = C.c_void_p
        record = bridge.record()
        tensor_lists = []
        def one_tensor(t):
            ptr = bridge.tensor(t, record)
            storage = (C.c_void_p*1)(ptr)
            record['keep'].append(storage)
            result = bridge.base.aclCreateTensorList(storage, 1)
            if not result:
                raise RuntimeError('aclCreateTensorList returned null')
            record['tensors'].remove(ptr)  # Ownership follows original binding's TensorList convention.
            tensor_lists.append(result)
            return result
        arguments = [one_tensor(x),one_tensor(weight)]+[None]*10+[0,-1,0,0,None,one_tensor(y),None,None]
        assert len(arguments) == 20
        address = C.cast(bridge.size, C.c_void_p).value
        mapping = next(line for line in Path('/proc/self/maps').read_text().splitlines()
                       if int(line.split()[0].split('-')[0],16) <= address < int(line.split()[0].split('-')[1],16))
        mapped_path = mapping.split(maxsplit=5)[5]
        assert Path(mapped_path).resolve() == host.resolve()
        assert digest(mapped_path) == json.loads((P/'PINS.json').read_text())[str(host)]
        report['symbol'] = dict(name='aclnnGroupedMatmulV5GetWorkspaceSize', address=address,
                                mapping=mapping, sha256=digest(mapped_path))
        report['before_query'] = sample()
        save()
        size, executor = C.c_uint64(), C.c_void_p()
        start = time.monotonic()
        report['query_calls'] = 1
        rc = bridge.size(*arguments, C.byref(size), C.byref(executor))
        report.update(return_code=rc, workspace_bytes=size.value, executor=executor.value,
                      query_seconds=time.monotonic()-start)
        # No bridge.call(), no launch API, and no allocation of the returned workspace.
        torch.npu.synchronize()
        report['after_query'] = sample()
        KEEP.extend((bridge, record, tensor_lists, arguments, x, weight, y, executor))
        report['lifetime'] = 'All descriptors/executor/storage remain live until this one-query process exits; no guessed executor destructor.'
        assert rc == 0 and executor.value, 'Original size query failed'
        report['status'] = 'workspace_query_passed_not_fullcase_verified'
    except BaseException:
        report['status'] = 'query_failed'
        report['error'] = traceback.format_exc()
        raise
    finally:
        report['finished'] = time.time()
        save()

if __name__ == '__main__':
    main()
