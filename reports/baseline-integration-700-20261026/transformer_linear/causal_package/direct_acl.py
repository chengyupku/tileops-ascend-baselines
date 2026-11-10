"""Private standard ACLNN transport. Import is stdlib-only; construction is guarded.
No GraphCall, native registration, helper DSO, kernel or arithmetic is implemented.
"""
import ctypes as C
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
P = C.c_void_p
I64P = C.POINTER(C.c_int64)
U64P = C.POINTER(C.c_uint64)
PP = C.POINTER(P)


def require_guard():
    metadata = json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text())
    owner = os.environ.get('GDN_CASE_OWNER')
    if (not owner or metadata.get('owner') != owner or
            metadata.get('phase') != 'running' or
            metadata.get('child', {}).get('pid') != os.getpgrp()):
        raise RuntimeError('An expressly authorized foreground npu-guard child is required')


class DirectACL:
    def __init__(self, op, torch, opapi, nnopbase):
        require_guard()
        if op not in ('RecurrentKda', 'ChunkGatedDeltaRule', 'CausalConv1dUpdate'):
            raise ValueError(op)
        self.torch, self.op = torch, op
        # Absolute paths are supplied by the pinned recipe, never a generic SDK API.
        self.base = C.CDLL(str(Path(nnopbase).resolve()), mode=C.RTLD_LOCAL)
        self.library = C.CDLL(str(Path(opapi).resolve()), mode=C.RTLD_LOCAL)
        self.records = []
        self.create = self.bind(self.base, 'aclCreateTensor',
            [I64P, C.c_uint64, C.c_int, I64P, C.c_int64, C.c_int,
             I64P, C.c_uint64, P], P)
        self.destroy = self.bind(self.base, 'aclDestroyTensor', [P])
        if op == 'CausalConv1dUpdate':
            signature = [P] * 9 + [C.c_char_p, C.c_int64, C.c_int64, P, U64P, PP]
        elif op == 'RecurrentKda':
            signature = ([P] * 11 + [C.c_char_p, C.c_double] + [C.c_bool] * 7 +
                         [C.c_double, C.c_bool, P, P, U64P, PP])
        else:
            signature = [P] * 7 + [C.c_float, P, P, U64P, PP]
        self.workspace = self.bind(self.library, 'aclnn' + op + 'GetWorkspaceSize', signature)
        self.launch = self.bind(self.library, 'aclnn' + op, [P, C.c_uint64, P, P])

    @staticmethod
    def bind(lib, name, arguments, result=C.c_int32):
        function = getattr(lib, name)
        function.argtypes, function.restype = arguments, result
        return function

    @staticmethod
    def check(code, stage):
        if code:
            raise RuntimeError(f'{stage} returned {code}; no automatic retry')

    def tensor(self, tensor, record):
        if tensor is None:
            return None
        if tensor.device.type not in ('npu', 'privateuseone') or not tensor.is_contiguous():
            raise ValueError('ACLNN transport requires contiguous device tensors')
        codes = {self.torch.float32: 0, self.torch.float16: 1,
                 self.torch.bfloat16: 27, self.torch.int32: 3, self.torch.int64: 9}
        dims = (C.c_int64 * tensor.ndim)(*tensor.shape)
        strides = (C.c_int64 * tensor.ndim)(*tensor.stride())
        # data_ptr already points at the view start. Treat the contiguous view as
        # a dense storage, with offset zero; original allocation stays alive.
        descriptor = self.create(dims, tensor.ndim, codes[tensor.dtype], strides,
                                 0, 2, dims, tensor.ndim, tensor.data_ptr())
        if not descriptor:
            raise RuntimeError('aclCreateTensor returned null')
        record['descriptors'].append(descriptor)
        record['keep'].extend((tensor, dims, strides))
        return descriptor

    def call(self, inputs, outputs, attributes):
        require_guard()
        record = {'descriptors': [], 'keep': []}
        self.records.append(record)
        try:
            ins = [self.tensor(t, record) for t in inputs]
            outs = [self.tensor(t, record) for t in outputs]
            size, executor = C.c_uint64(), P()
            record['keep'].extend((size, executor))
            self.check(self.workspace(*ins, *attributes, *outs, C.byref(size), C.byref(executor)),
                       'aclnn' + self.op + 'GetWorkspaceSize')
            if not executor.value:
                raise RuntimeError('null ACLNN executor')
            # Full returned size is allocated. A resource ceiling rejects the run,
            # never truncates the workspace or changes the case.
            if size.value > 1024 ** 3:
                raise RuntimeError(f'workspace {size.value} exceeds proposed 1 GiB allocation envelope')
            workspace = self.torch.empty((size.value,), dtype=self.torch.uint8,
                                          device=outputs[0].device) if size.value else None
            record['keep'].append(workspace)
            stream = self.torch.npu.current_stream().npu_stream
            self.check(self.launch(None if workspace is None else workspace.data_ptr(),
                                   size.value, executor, stream), 'aclnn' + self.op)
        finally:
            # One-shot executor ownership follows the original ACLNN example:
            # launch consumes it; do not double-destroy after execution.
            # A phase-one failure/ceiling rejection ends the process, no retry.
            self.cleanup()

    def cleanup(self):
        self.torch.npu.synchronize()
        while self.records:
            record = self.records.pop()
            for descriptor in record['descriptors']:
                self.check(self.destroy(descriptor), 'aclDestroyTensor')
