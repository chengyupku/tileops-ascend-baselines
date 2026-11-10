"""Layout/API adapters only. Device arithmetic belongs to the original FLA source."""
from pathlib import Path


def cumsum(case):
    import importlib.util
    import sys
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, LIBRARIES, source_hashes

    source = LIBRARIES / "flash-linear-attention-npu/fla/ops/triton/triton_core/cumsum.py"
    spec = importlib.util.spec_from_file_location("_allpairs_fla_cumsum", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    dim = case.params.get("dim", -1)

    def run(x):
        data = x.movedim(dim, -1).contiguous()
        size = data.shape[-1]
        extent = 1 << (size - 1).bit_length()
        padded = F.pad(data, (0, extent - size)).reshape(-1, extent, 1)
        out = module.chunk_local_cumsum(padded, chunk_size=extent, output_dtype=x.dtype)
        return out[..., 0][:, :size].reshape(data.shape).movedim(-1, dim).contiguous()

    return Candidate(Implementation(run), "fla.ops.triton.triton_core.cumsum.chunk_local_cumsum",
                     "Move cumulative axis last; each independent row is one native chunk; zero-pad to power of two then crop. Original host/kernel/schedule unchanged.",
                     source_hashes(Path(__file__), source))
