"""Ascend sinusoidal positional encoding with reusable column divisors."""
import functools
from typing import ClassVar
import torch
from tileops.kernels.elementwise.call_spec import SinusoidalFwdInterface
from tileops.kernels.kernel_base import Kernel
try:
    import tilelang
    import tilelang.ascend.language as T
except ImportError:
    T = None

if T is not None:
    @T.macro
    def _divisors(divisor, cols: int, col0, d_model: int):
        with T.SimtVF(threads=cols):
            for c in T.Parallel(cols):
                exponent = T.cast(col0 + c, 'float32') * T.cast(2., 'float32') / T.cast(d_model, 'float32')
                divisor[c] = T.pow(T.cast(10000., 'float32'), exponent)

    @T.macro
    def _encode(divisor, output, rows: int, cols: int, row0, dtype: str):
        with T.SimtVF(threads=min(2048, rows * cols)):
            for chunk in T.serial((rows * cols + min(2048, rows * cols) - 1) // min(2048, rows * cols)):
                for lane in T.Parallel(min(2048, rows * cols)):
                    pair = chunk * min(2048, rows * cols) + lane
                    if pair < rows * cols:
                        r = pair // cols
                        c = pair % cols
                        angle = T.cast(row0 + r, 'float32') / divisor[c]
                        output[r, 2 * c] = T.cast(T.sin(angle), dtype)
                        output[r, 2 * c + 1] = T.cast(T.cos(angle), dtype)

    @functools.lru_cache(maxsize=128)
    def build(seq_len, d_model, dtype):
        half = d_model // 2
        rows, cols, stages = min(16, seq_len), min(256, half), 2
        row_tiles, col_tiles = (seq_len + rows - 1) // rows, (half + cols - 1) // cols
        total = row_tiles * col_tiles
        grid = min(64, total)
        @T.prim_func
        def main(Y: T.Tensor((seq_len, d_model), dtype)):
            with T.Kernel(grid) as bx:
                divisor = T.alloc_shared((cols,), 'float32')
                output = T.alloc_shared((rows, 2 * cols), dtype)
                T.annotate_buffer_versions({divisor: stages, output: stages})
                for step in T.Pipelined((total + grid - 1) // grid, num_stages=stages):
                    block = step * grid + bx
                    if block < total:
                        row0 = (block // col_tiles) * rows
                        col0 = (block % col_tiles) * cols
                        valid_rows = T.min(rows, seq_len - row0)
                        valid_cols = T.min(cols, half - col0)
                        _divisors(divisor, cols, col0, d_model)
                        _encode(divisor, output, rows, cols, row0, dtype)
                        T.copy(output[:valid_rows, :2 * valid_cols], Y[row0:row0 + valid_rows, 2 * col0:2 * (col0 + valid_cols)], l2_cache_ctrl=0)
        return tilelang.compile(main, target='ascend', out_idx=[0])

class AscendSinusoidalKernel(Kernel, SinusoidalFwdInterface):
    devices: ClassVar[frozenset[str]] = frozenset({'npu'})
    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call.seq_len, call.d_model, call.dtype, call.device)
    def __init__(self, seq_len, d_model, dtype, device=None):
        super().__init__()
        if seq_len <= 0 or d_model <= 0 or d_model % 2:
            raise ValueError('Sinusoidal requires positive length and positive even model dimension')
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(f'Unsupported sinusoidal dtype: {dtype}')
        self.seq_len, self.d_model, self.dtype = seq_len, d_model, dtype
        self.device = torch.device(device) if device is not None else torch.device('npu', torch.npu.current_device())
        self.compiled = build(seq_len, d_model, str(dtype).removeprefix('torch.'))
    def forward(self):
        with torch.npu.device(self.device):
            return self.compiled()

if T is not None:
    from tileops.backend import register_kernel_type
    register_kernel_type('SinusoidalFwdOp', 'ascend', AscendSinusoidalKernel)
