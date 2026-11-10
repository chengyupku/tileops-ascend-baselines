"""Chunkwise GLA forward using the independently validated Ascend pipelines."""
from typing import ClassVar

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.linear_attention import GLAChunkFwdInterface

from .gla import T, build
from .gla_chunked import build_chunked


class AscendGLAChunkFwdKernel(Kernel, GLAChunkFwdInterface):
    devices: ClassVar[frozenset[str]] = frozenset({'npu'})

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call)

    def __init__(self, call):
        super().__init__()
        self.call = call

    def forward(self, q, k, v, g, initial_state=None):
        c = self.call
        scale = c.scale if c.scale > 0 else c.dim_k ** -0.5
        q, k, v, g = (x.contiguous() for x in (q, k, v, g))
        seeded = initial_state is not None
        state = initial_state.contiguous() if seeded else torch.empty(
            (c.batch, c.heads, c.dim_k, c.dim_v), dtype=torch.float32, device=q.device
        )
        args = (c.batch, c.seq_len, c.heads, c.dim_k, c.dim_v,
                str(c.dtype).removeprefix('torch.'), scale, seeded)
        if c.dtype in (torch.float16, torch.bfloat16) and c.dim_k in (64, 128) and c.dim_v in (64, 128):
            self.compiled = build_chunked(*args)
        else:
            # FP32 and arbitrary widths retain FP32 state throughout the recurrence.
            self.compiled = build(*args, ())
        return self.compiled(q, k, v, g, state)


if T is not None:
    from tileops.backend import register_kernel_type
    register_kernel_type('GLAChunkFwdOp', 'ascend', AscendGLAChunkFwdKernel)
