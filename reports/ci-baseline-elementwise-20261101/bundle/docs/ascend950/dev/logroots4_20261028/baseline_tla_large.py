"""Recheck original large TLA unary kernels with a 128-row specialization."""
import dataclasses
import hashlib
from math import prod
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / 'TileOPs'), str(ROOT / 'TileOPs/src'),
               str(HERE.with_name('arithmetic9_20261027'))]
from benchmarks.ascend_all_pairs.tla import adapters, native
original = adapters.elementwise


def elementwise(op, case):
    if op not in {'LogFwdOp', 'SqrtFwdOp', 'RsqrtFwdOp'} or prod(case.params['output'][0]) != 256 * 1024 * 1024:
        return original(op, case)
    adapters._check(op, case)
    import torch
    shape, output_dtype = adapters._output(case)
    size = prod(shape)
    rows, cols = size // 64, 64
    name, arity = adapters.DIRECT[op]
    assert arity == 1 and rows % 128 == 0
    module = native.load(adapters.ELEMENTWISE, name, 'pass_configs')
    kernel = native.compile_builder(module, name, rows, cols, 128, 64,
                                    dtype='float32', out_idx=[-1])

    def run(x):
        packed = adapters._flat_pack(x, shape, rows, cols, torch.float32, fill=1)
        return kernel(packed).reshape(shape).to(output_dtype)

    candidate = native.candidate(
        run, f'{adapters.ELEMENTWISE}:{name}(block_M=128,block_N=64)',
        'Unmodified original builder and unary tile operation. Use its 128x64 specialization '
        'to reduce the 256M launch grid from 65536 to 32768 blocks; exact input widening and '
        'output cast remain inside each timed call. Numerical verification is mandatory.',
        adapters.ELEMENTWISE, adapters.TILE_API)
    return dataclasses.replace(candidate, source_sha256=candidate.source_sha256 | {
        str(Path(__file__)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})


adapters.elementwise = elementwise
runpy.run_path(str(HERE / 'baselines.py'), run_name='__main__')
