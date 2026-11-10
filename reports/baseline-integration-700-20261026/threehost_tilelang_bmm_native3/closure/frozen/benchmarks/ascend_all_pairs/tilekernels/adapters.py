"""Layout/API adapters only; all arithmetic belongs to original TileKernels."""
from benchmarks.ascend_baselines import Candidate, LIBRARIES, local_function, source_hashes


def silu_and_mul(case):
    import torch
    from benchmarks.api import Implementation
    fn = local_function("TileKernels", ".", "tile_kernels.quant.swiglu_forward_kernel",
                        "swiglu_forward", "tile_kernels/quant/swiglu_forward_kernel.py")
    dtype = case.inputs[0].dtype
    def run(x):
        # FP16 values widen losslessly; native FP32 output is restored to contract dtype.
        return fn(x.contiguous().to(torch.float32), "fp32").to(dtype)
    root = LIBRARIES / "TileKernels/tile_kernels/quant"
    return Candidate(Implementation(run), "tile_kernels.quant.swiglu_forward_kernel.swiglu_forward",
                     "Contiguous 2D input; exact FP16/BF16 widening; native FP32 SwiGLU; restore output dtype on every call.",
                     source_hashes(root / "swiglu_forward_kernel.py", root / "swiglu_forward_asc.py"))
