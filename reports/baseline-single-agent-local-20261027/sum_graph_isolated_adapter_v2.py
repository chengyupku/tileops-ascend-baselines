"""Use the existing full graph API with a ReduceSum-only original deployment."""
from dataclasses import replace
import json
import os
from pathlib import Path


def factory(case, output, pins):
    import torch
    from benchmarks.api import Implementation
    import importlib.util
    from benchmarks.ascend_all_pairs.ops_math import graph as original_graph
    spec = importlib.util.spec_from_file_location(
        "benchmarks.ascend_all_pairs.ops_math._single_agent_sum_private", original_graph.__file__)
    graph = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(graph)
    assert case.id == "mlp-intermediate-float32"
    # Deployment selection only. Keep original graph binding, axes, dtype,
    # source, tiling, kernel and checker unchanged.
    graph.VENDOR = Path(os.environ["ASCEND_CLASSIC_MATH_VENDOR"])
    assert graph.deployment_hashes.cache_info().currsize == 0
    candidate = graph.make(case, "SumFwdOp")

    def run(x):
        print("ISOLATED_ORIGINAL_REDUCESUM_GRAPH_ENTERED", flush=True)
        result = candidate.implementation.run(x)
        torch.npu.synchronize()
        (output / "native.json").write_text(json.dumps({
            "api": candidate.api, "vendor": str(graph.VENDOR),
            "input": list(x.shape), "output": list(result.shape),
            "synchronized": True}, indent=2))
        (output / "native_maps.txt").write_text(Path("/proc/self/maps").read_text())
        return result

    return replace(candidate, implementation=Implementation(run),
                   source_sha256=candidate.source_sha256 | pins,
                   adaptation="Original complete graph ReduceSum, axis -1, FP32, unchanged "
                   "layout; isolated original deployment excludes unrelated custom operators.")
