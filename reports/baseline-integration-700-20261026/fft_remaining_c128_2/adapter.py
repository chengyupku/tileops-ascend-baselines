"""Private exact-case binding: original aclStft with explicit complex unity window."""
import ctypes as C
from pathlib import Path
from benchmarks.ascend_baselines import Candidate, LIBRARIES, source_hashes
from benchmarks.ascend_math_more import _MathCall


def make(case):
    import torch
    from benchmarks.api import Implementation
    shapes = {'throughput-4k-complex128': (64,4096), 'scientific-1m-complex128': (4,1048576)}
    assert case.id in shapes
    shape = shapes[case.id]
    assert case._call.signature.name == 'FFTC2CFwdOp'
    x, = case.inputs
    assert tuple(x.shape) == shape and x.dtype == torch.complex128
    binding = _MathCall('aclStft', [C.c_void_p] * 3 + [C.c_int64] * 3 + [C.c_bool] * 3)
    binding.codes[torch.complex128] = 17
    n = shape[-1]
    # Constant construction only: no input arithmetic, FFT, or alternate kernel.
    window = torch.ones((n,), dtype=x.dtype, device='cpu').to(x.device)

    def run(x):
        assert tuple(x.shape) == shape and x.dtype == window.dtype
        assert x.device == window.device and x.is_contiguous() and window.is_contiguous()
        out = torch.empty((*shape, 1), dtype=x.dtype, device=x.device)
        record = binding.record()
        binding.call([binding.tensor(x, record), binding.tensor(window, record),
                      binding.tensor(out, record), n, n, n, False, False, True],
                     record, x.device)
        return out.reshape(shape)

    hashes = binding.hashes(LIBRARIES / 'ops-math/math/stft')
    hashes.update(source_hashes(Path(__file__)))
    return Candidate(Implementation(run), 'ops-math::aclStft (explicit complex unity window)',
                     'N=hop=win=shape[-1]; complex128 ACL dtype 17; same-dtype/device unity window [N]; full [B,N,1] output reshaped [B,N].',
                     hashes, limitations=('Original complete API system CANN AiCPU STFT route; not custom vendor STFT; correctness only.',),
                     timing_mode='aicpu', cleanup=binding.cleanup)
