"""Shape/layout bindings for additional unmodified upstream Ascend examples."""
from pathlib import Path

from benchmarks.ascend_baselines import (
    Candidate, LIBRARIES, UnsupportedCase, _add_layout, local_function, source_hashes,
)
from benchmarks.ascend_tilelang_baselines import _runtime, _runtime_hashes


def tilelang_add(case):
    import torch.nn.functional as F
    from benchmarks.api import Implementation

    tilelang = _runtime()
    x, _, shape, rows, hidden, pack = _add_layout(case)
    size = rows * hidden
    native_size = ((size + 8192 * 64 - 1) // (8192 * 64)) * (8192 * 64)
    host = "examples/ascend/example_vecadd.py"
    builder = local_function("tilelang", "examples/ascend", "example_vecadd", "vector_add", host)
    kernel = tilelang.compile(builder(native_size, mode="simt"), target="ascend", out_idx=-1)

    def run(x, y):
        # The native API uses FP32. Widen BF16 exactly, then round the sum once
        # back to the frozen output dtype; all casts belong to this invocation.
        a = F.pad(pack(x).flatten().float(), (0, native_size - size))
        b = F.pad(pack(y).flatten().float(), (0, native_size - size))
        return kernel(a, b)[:size].to(x.dtype).view(shape)

    return Candidate(
        Implementation(run), "tilelang::example_vecadd.vector_add(simt)",
        "Exact FP16/BF16 widening, broadcast and independent tail padding to complete native tiles; crop/cast output; original pipeline unchanged.",
        source_hashes(Path(__file__), LIBRARIES / "tilelang" / host) | _runtime_hashes(tilelang),
    )


def tilelang_fp8(case):
    import torch.nn.functional as F
    from benchmarks.api import Implementation

    tilelang = _runtime()
    x, = case.inputs
    if x.shape[-1] != 64 or x.numel() // 64 % 32:
        raise UnsupportedCase("This sample pads independent 64-value groups to the original 128-value API.")
    host = "examples/ascend/example_per_token_cast_to_fp8.py"
    builder = local_function(
        "tilelang", "examples/ascend", "example_per_token_cast_to_fp8",
        "per_token_cast_to_fp8", host)
    kernel = builder(x.numel() // 64, 128, mode="simt")
    shape = tuple(x.shape)

    def run(x):
        # Padding each group with zero preserves absmax and the original clamp.
        padded = F.pad(x.float().reshape(-1, 64), (0, 64))
        quant, scale = kernel(padded)
        return scale.reshape(shape[:-1]), quant[:, :64].contiguous().reshape(shape)

    return Candidate(
        Implementation(run), "tilelang::per_token_cast_to_fp8(simt)",
        "Exact FP16 widening; per-group zero pad64->128 and output crop; original quantization/scale and schedule.",
        source_hashes(Path(__file__), LIBRARIES / "tilelang" / host) | _runtime_hashes(tilelang),
    )


def tilelang_gqa(case):
    import ast
    import hashlib
    import importlib.util
    import sys
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import WORKSPACE

    tilelang = _runtime()
    q, k, v, *extra = case.inputs
    batch, sq, hq, dim = q.shape
    _, sk, hk, _ = k.shape
    p = case.params
    group = hq // hk
    if (any(t is not None for t in extra)
            or q.dtype != torch.float16 or k.dtype != q.dtype or v.dtype != q.dtype
            or sq != 1 or dim != 128 or hq % hk or group > 128 or sk % 128
            or p.get("softcap") is not None or p.get("pos_encoding_mode", "none") != "none"
            or p.get("window_size_left", -1) != -1 or p.get("window_size_right", -1) != -1):
        raise UnsupportedCase("This first sample adapts full-context FP16 decode to independent head groups.")
    original = LIBRARIES / "tilelang/examples/ascend/flash_attention/core.py"
    # Both formats occupy 16 bits. Change only dtype string constants, including
    # native cast/packing types. Every loop, memory layout and schedule stays
    # byte-for-byte identical. Never quantize the FP16 inputs to BF16.
    source = original.read_text()
    adapted = source.replace("bfloat16", "float16")
    before, after = ast.parse(source), ast.parse(adapted)
    class DtypeOnly(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str):
                node.value = node.value.replace("bfloat16", "float16")
            return node
    if ast.dump(DtypeOnly().visit(before)) != ast.dump(after):
        raise RuntimeError("Attention dtype adaptation altered more than string constants.")
    key = hashlib.sha256(adapted.encode()).hexdigest()
    cached = WORKSPACE / ".cache/ascend-classic" / ("tilelang-gqa-fp16-" + key[:20])
    cached.mkdir(parents=True, exist_ok=True)
    path = cached / "core.py"
    if path.exists() and path.read_text() != adapted:
        raise RuntimeError("Generated dtype-only source does not match its key.")
    path.write_text(adapted)
    name = "_classic_tilelang_gqa_" + key[:20]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    program = module.flash_attention_fwd(
        128, sk, head_dim=dim, out_dtype="float16", softmax_scale=p.get("sm_scale"), num_blocks=1)
    kernel = tilelang.compile(program, target="ascend", out_idx=-1)

    def run(q, k, v, *extra):
        out = torch.empty_like(q)
        for b in range(batch):
            for head in range(hk):
                queries = q[b, 0, head * group:(head + 1) * group]
                queries = F.pad(queries, (0, 0, 0, 128 - group))
                values = kernel(queries, k[b, :, head].contiguous(), v[b, :, head].contiguous())
                out[b, 0, head * group:(head + 1) * group].copy_(values[:group])
        return out

    return Candidate(
        Implementation(run), "tilelang::flash_attention_fwd (dtype-only FP16 instantiation)",
        "Only dtype literals change; per-batch/KV-head API calls, query padding and output crop; original schedule.",
        source_hashes(Path(__file__), original, path) | _runtime_hashes(tilelang),
        limitations=("Host loop launches each KV-head group separately; full adapter cost belongs in timing.",),
    )
