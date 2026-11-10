"""Thin interface to the original complete TileLang BF16 batched GEMM API."""
import hashlib
import json
import sys
from pathlib import Path

W = Path('/opt/tileops-all-pairs/workspace')

def factory(case, out):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate
    sys.path.insert(0, str(W / 'tilelang/examples/ascend'))
    from deepgemm.api import bf16_bmm
    src = W / 'tilelang/examples/ascend/deepgemm/api.py'
    assert Path(bf16_bmm.__code__.co_filename).resolve() == src.resolve()
    assert hashlib.sha256(src.read_bytes()).hexdigest() == '457af5e2d4ede9c67e39ec3aa0f9750b2e022e87f0c93a9b5f94297a6c83b4e4'
    assert case.id in {'mha-decode-pv-bfloat16', 'moe-prefill-bfloat16'}
    a, b = case.inputs
    assert a.dtype == b.dtype == torch.bfloat16 and a.ndim == b.ndim == 3
    calls = []
    def run(a, b):
        d = torch.empty((a.shape[0], a.shape[1], b.shape[2]), dtype=a.dtype, device=a.device)
        bf16_bmm(a, b.transpose(-2, -1), d)
        torch.npu.synchronize()
        calls.append({'api': 'deepgemm.api.bf16_bmm', 'one_complete_call': True,
                      'synchronized': True, 'a_shape': list(a.shape), 'b_shape': list(b.shape)})
        (out / 'calls.json').write_text(json.dumps(calls, indent=2))
        return d
    pins = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(src.parent.rglob('*.py'))}
    pins[str(Path(__file__).resolve())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return Candidate(Implementation(run), 'tilelang::deepgemm.api.bf16_bmm',
                     'Original complete batched API; B transpose view and output allocation only; no per-batch composition or precision conversion.',
                     pins, timing_mode='host_roundtrip')
