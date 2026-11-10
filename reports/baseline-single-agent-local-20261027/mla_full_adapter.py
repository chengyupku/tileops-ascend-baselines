"""Original complete MLA API: concatenate rope fields and enumerate all keys."""
import hashlib
import json
from pathlib import Path


def factory(case, native, tilelang, output, identity):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate
    p = case.params
    batch, heads, dim = p["q"][0]
    length, groups = p["k"][0][1:3]
    tail = p["q_pe"][0][-1]
    source = "examples/sparse_flash_attention/example_sparse_flash_attn_dynamic_shape.py"
    module = native.load(source, "sparse_attention_fwd")
    module.DataType = tilelang.DataType
    builder = module.sparse_attention_fwd
    config = builder.__jit_impl__
    kernel = tilelang.compile(
        builder.__wrapped__(heads, dim, tail, length, 1,
                            kv_group=groups, dtype=p["q"][1]),
        out_idx=config.out_idx, workspace_idx=config.workspace_idx,
        target="pto", platform="A5", pass_configs=config.pass_configs,
        execution_backend=config.execution_backend, target_host=config.target_host,
        compile_flags=config.compile_flags)
    pins = dict(identity)
    for path in (Path(__file__), native.VENDOR / source):
        pins[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / "binding.json").write_text(json.dumps({
        "api": source + ":sparse_attention_fwd", "params": p,
        "original_pass_configs": str(config.pass_configs),
        "original_workspace_indices": config.workspace_idx,
        "native_calls": 1, "full_batch": batch, "full_key_extent": length}, indent=2))

    def run(q, q_pe, k, k_pe):
        query = torch.cat((q, q_pe), -1).unsqueeze(1).contiguous()
        key = torch.cat((k, k_pe), -1).contiguous()
        indices = torch.arange(length, dtype=torch.int32, device=q.device)
        indices = indices.reshape(1, 1, 1, length).expand(batch, 1, groups, length).contiguous()
        torch.npu.synchronize()
        (output / "PRE_SYNC_COMPLETED.json").write_text('{"synchronized": true}\n')
        actual = Path(kernel.adapter.lib._name).resolve()
        code = output / "actual_kernel.cu"
        code.write_text(kernel.get_kernel_source())
        for path in (actual, code):
            pins[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        (output / "ACTUAL_LAUNCH.json").write_text(json.dumps({
            "library": str(actual), "sha256": pins[str(actual)],
            "target": str(kernel.target), "native_calls": 1}, indent=2))
        result = kernel(query, key, indices)
        (output / "HOST_RETURNED.json").write_text('{"returned": true}\n')
        torch.npu.synchronize()
        (output / "DEVICE_SYNCHRONIZED.json").write_text('{"synchronized": true}\n')
        return result[:, 0].contiguous()

    return Candidate(Implementation(run), "tilelang-ascend::" + source + ":sparse_attention_fwd",
                     "One complete original MLA call, with rope concatenation and all-key indices. "
                     "All copies inside run; original kernel and precision unchanged.",
                     pins, timing_mode="host_roundtrip")
