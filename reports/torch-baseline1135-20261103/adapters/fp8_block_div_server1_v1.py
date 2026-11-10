import torch
from workloads.quantization.quantize import _pad_to_blocks, _BLOCK
def fp8_quant_per_block(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``FP8QuantPerBlockFwdOp``'s reference: one scale per 128x128 tile."""
    n, k = w.shape
    wf = w.float()
    padded = _pad_to_blocks(wf, rows=True)
    tiles = padded.view(padded.shape[0] // _BLOCK, _BLOCK, padded.shape[1] // _BLOCK, _BLOCK)
    amax = tiles.abs().amax(dim=(1, 3))
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    scale = torch.where(amax > 0, amax / torch.full_like(amax, fp8_max), torch.ones_like(amax))
    full = scale.repeat_interleave(_BLOCK, 0).repeat_interleave(_BLOCK, 1)[:n, :k]
    q = (wf / full).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    return q, scale

