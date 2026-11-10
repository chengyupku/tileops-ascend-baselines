"""Use the original complete checker with a private top-level candidate."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

P = Path(__file__).resolve().parent
W = Path('/opt/tileops-all-pairs/workspace')
key = sys.argv[1]
assert key in ('prefill-4x1k-float16', 'prefill-8x1k-float16', 'prefill-ragged-bfloat16')
A = P / key
A.mkdir(exist_ok=True)
sys.path[:0] = [str(P), str(W/'TileOPs'), str(W/'TileOPs/src'), str(W)]
pins = json.loads((P/'DEPLOYMENT_PINS.json').read_text())
for path, digest in pins.items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
for name in ('runner.py', 'adapter.py', 'direct_api.py'):
    pins[str(P/name)] = hashlib.sha256((P/name).read_bytes()).hexdigest()

from benchmarks.ascend_all_pairs import registry
from benchmarks.ascend_baselines import Candidate
from benchmarks.api import Implementation
import adapter
import direct_api

def factory(case, library):
    assert library == 'ops-transformer' and case.id == key
    assert case._call.signature.name == 'NSACompressedVarlenFwdOp'
    def run(q, k, v, offsets, chunk_offsets, token_indices):
        import torch
        return adapter.compressed(torch, direct_api.flash_attn,
            direct_api.flash_attn_metadata, q, k, v, offsets,
            chunk_offsets, token_indices, bs=case.params['bs'],
            scale=case.params['scale'])
    return lambda: Candidate(Implementation(run),
        'Original FlashAttn + FlashAttnMetadata with packed prefix batches',
        'One complete original attention call. Runtime packing, copies, empty-prefix zero output, LSE layout/dtype conversion included.',
        pins, timing_mode='host_sync')

registry.candidate_factory = factory
sys.argv = [str(W/'TileOPs/scripts/ascend_pair_check.py'), '--op',
    'NSACompressedVarlenFwdOp', '--case-id', key, '--library',
    'ops-transformer', '--output', str(A/'checker.json')]
try:
    runpy.run_path(sys.argv[0], run_name='__main__')
finally:
    torch = sys.modules.get('torch')
    if torch is not None and torch.npu.is_initialized():
        torch.npu.synchronize()
    (A/'native.json').write_text(json.dumps({'events': direct_api.EVENTS,
        'pins': pins, 'post_drift': [path for path, h in pins.items()
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != h]}, indent=2))
    (A/'maps').write_text(Path('/proc/self/maps').read_text())
