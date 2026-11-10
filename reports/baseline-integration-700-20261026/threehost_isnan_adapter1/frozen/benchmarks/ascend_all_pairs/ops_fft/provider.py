"""Original opsfft_fft route with AArch64 deployment and native call receipts.

The API consumes host pointers. Every invocation retains D2H, original native
execution, and H2D. No CPU FFT is used by this candidate.
"""
from math import prod
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[4]
BINARY = WORKSPACE / '.cache/all-pairs/remote_ops_fft/build-910/libcann_ops_fft.so.9.0.0'
SYMBOLS = ('aclfftPlan1d', 'aclfftSetStream', 'aclfftExecC2C', 'aclfftDestroy')


def fft(case):
    import ctypes as C
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, UnsupportedCase, _owned_native, source_hashes

    (x,) = case.inputs
    if x.dtype != torch.complex64:
        raise UnsupportedCase('ops-fft aclfftExecZ2Z is an ACLFFT_NOT_IMPLEMENTED stub; complex128 is unsupported.')
    n, batch = x.shape[-1], prod(x.shape[:-1])
    if n <= 1 or n > 1 << 27 or n & (n - 1):
        raise UnsupportedCase('This adapter uses the inspected power-of-two C2C path.')
    lib = _owned_native(BINARY, SYMBOLS)
    lib.aclfftPlan1d.argtypes = [C.POINTER(C.c_void_p), C.c_int, C.c_int, C.c_int, C.c_int]
    lib.aclfftSetStream.argtypes = [C.c_void_p, C.c_void_p]
    lib.aclfftExecC2C.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_int]
    lib.aclfftDestroy.argtypes = [C.c_void_p]
    for name in SYMBOLS:
        getattr(lib, name).restype = C.c_int
    calls = []

    def check(name, rc):
        calls.append({'api': name, 'returncode': rc})
        if rc:
            raise RuntimeError(f'ops-fft {name} native status={rc}')

    plan = C.c_void_p()
    check('aclfftPlan1d', lib.aclfftPlan1d(C.byref(plan), n, 0x29, batch, 0))

    def run(x):
        host_in = x.detach().cpu().contiguous()
        host_out = torch.empty_like(host_in)
        check('aclfftSetStream', lib.aclfftSetStream(plan, torch.npu.current_stream().npu_stream))
        check('aclfftExecC2C', lib.aclfftExecC2C(plan, host_in.data_ptr(), host_out.data_ptr(), -1))
        return host_out.to(x.device)

    # Passive evidence for the task observer, without replacing the ctypes call.
    run.native_calls = calls
    run.native_library = lib
    run.native_plan = {'n': n, 'batch': batch, 'type': 0x29, 'direction': -1, 'shape': tuple(x.shape)}
    root = WORKSPACE / 'ascend_baselines/ops-fft'
    return Candidate(
        Implementation(run), 'aclfftPlan1d(ACLFFT_C2C)+aclfftExecC2C(ACLFFT_FORWARD)',
        'Original plan/execute API; contiguous host staging and transfers inside every call.',
        source_hashes(BINARY, Path(__file__), root/'include/cann_ops_fft.h',
                      root/'src/fft1_d/arch35/fft1_d_exec_api.cpp',
                      root/'src/fft1_d/arch35/fft/fft1_d_c2c.cpp',
                      root/'src/fft1_d/arch35/fft/fft1_d_c2c_kernel.h'),
        timing_mode='host_roundtrip',
        cleanup=lambda: check('aclfftDestroy', lib.aclfftDestroy(plan)),
    )
