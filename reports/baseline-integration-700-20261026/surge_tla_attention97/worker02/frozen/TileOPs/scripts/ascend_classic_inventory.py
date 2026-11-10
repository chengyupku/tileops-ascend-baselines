#!/usr/bin/env python3
"""Classify the local manifest and freeze the representative Ascend baseline scope.

CPU only: read YAML and the existing case inventory, without importing TileOPs,
Torch, or any baseline runtime. A missing or changed manifest fails explicitly.
"""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
WEBSITE = "https://tile-ai.github.io/TileOPs.github.io/zh/benchmarks/"
CATEGORIES = [
    ("elementwise", "Elementwise", ["elementwise"], "AddFwdOp", 70),
    ("rope", "RoPE", ["rope"], "RoPEFwdOp", 5),
    ("reduction", "Reduction", ["reduction"], "AmaxFwdOp", 19),
    ("normalization", "Normalization", ["norm"], "RMSNormFwdOp", 12),
    ("conv_pool", "Conv & Pool", ["convolution", "pool"], "Conv2dFwdOp", 16),
    ("gemm", "GEMM", ["gemm"], "GemmFwdOp", 8),
    ("quantization", "Quantization & Dequantization", ["quantization"], "FP8QuantFwdOp", 10),
    ("attention", "Attention", ["attention"], "GQADenseFwdOp", 20),
    ("moe", "MoE", ["moe"], "MoEPrePermuteFwdOp", 12),
    ("sampling", "Sampling", ["sampling"], "TopKMaskFwdOp", 6),
    ("linear_attention", "Linear Attention", ["linear_attention"], "DeltaNetFwdOp", 10),
    ("ssm", "SSM", ["mamba"], "SSDChunkStateFwdOp", 10),
    ("other", "Other", ["fft", "sequence_modeling", "transform"], "FFTC2CFwdOp", 7),
]

# Phase one deliberately freezes one ordinary workload per category BEFORE
# library validation. Phase two retains every case; failed pilots are not
# silently replaced with a more convenient shape or dtype.
PILOT_CASES = {
    "AddFwdOp": "hidden-state-prefill-bfloat16",
    "RoPEFwdOp": "neox-1d-2k-float16",
    "AmaxFwdOp": "hidden-state-float16",
    "RMSNormFwdOp": "llama-8b-prefill-bfloat16",
    "Conv2dFwdOp": "resnet-3x3-float16",
    "GemmFwdOp": "square-1k-bfloat16",
    "FP8QuantFwdOp": "kv-index-8k-float16",
    "GQADenseFwdOp": "llama-8b-4k-float16",
    "MoEPrePermuteFwdOp": "decode-t32-bfloat16",
    "TopKMaskFwdOp": "llama-8b-b1-bfloat16",
    "DeltaNetFwdOp": "prefill-2k-bfloat16",
    "SSDChunkStateFwdOp": "mamba2-780m-bfloat16-float32",
    "FFTC2CFwdOp": "latency-4k-complex64",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory():
    run = WORKSPACE / "docs/ascend950/results/remeasure_20261018_v2"
    frozen = json.loads((run / "sources.json").read_text())["sha256"]
    ops, origins, hashes = {}, {}, {}
    for path in sorted((ROOT / "src/tileops/manifest/spec").glob("*.yaml")):
        if path.name == "types.yaml":
            continue
        relative = str(path.relative_to(WORKSPACE))
        digest = sha256(path)
        if frozen.get(relative) != digest:
            raise ValueError(f"Case inventory needs regeneration: manifest changed: {relative}")
        hashes[relative] = digest
        for name, entry in yaml.safe_load(path.read_text()).items():
            if name in ops:
                raise ValueError(f"Duplicate manifest op: {name}")
            ops[name], origins[name] = entry, str(path.relative_to(ROOT))
    if len(ops) != 205:
        raise ValueError(f"Expected the requested 205-op scope, got {len(ops)}")
    plan = json.loads((run / "plan.json").read_text())
    cases = plan["expected_cases"]
    family_category = {f: key for key, _, families, _, _ in CATEGORIES for f in families}
    counts = Counter(family_category[e["family"]] for e in ops.values())
    rows, categories, representatives = [], [], []
    for key, title, families, representative, expected_count in CATEGORIES:
        if counts[key] != expected_count or ops[representative]["family"] not in families:
            raise ValueError(f"Category mapping changed: {key}")
        selected_cases = [c for c in cases if c["op"] == representative]
        if not selected_cases:
            raise ValueError(f"Representative has no inventoried cases: {representative}")
        categories.append(dict(
            category=key, title=title, families=families, op_count=counts[key],
            representative=representative, case_count=len(selected_cases),
        ))
        representatives.extend(selected_cases)
    selected_ops = {r["representative"] for r in categories}
    per_op_cases = Counter(c["op"] for c in cases)
    for name, entry in sorted(ops.items()):
        rows.append(dict(
            category=family_category[entry["family"]], family=entry["family"],
            op=name, representative=name in selected_ops,
            inventoried_case_count=per_op_cases[name], manifest=origins[name],
        ))
    libraries = sorted(p.name for p in (WORKSPACE / "ascend_baselines").iterdir()
                       if p.is_dir() and p.name != "ascend950_baselines")
    if len(libraries) != 17 or not {"vllm-ascend", "sgl-kernel-npu"} <= set(libraries):
        raise ValueError("Independent local library inventory changed")
    pilot = [c for c in representatives if PILOT_CASES[c["op"]] == c["case_id"]]
    if len(pilot) != 13:
        raise ValueError("Every representative must have exactly one frozen pilot case.")
    return dict(
        schema="ascend-classic-baseline-inventory/v1",
        category_source=WEBSITE,
        baseline_layout_source="https://github.com/tile-ai/TileOPs/tree/main/benchmarks",
        scope="One representative per website category; every local inventoried case; all local libraries.",
        website_snapshot_case_count=2207,
        local_factory_case_count=len(cases),
        version_note="Website cases are newer; local manifest hashes match the frozen local case inventory.",
        categories=categories, operators=rows, libraries=libraries,
        active_phase="one_case_per_operator",
        pilot_cases=pilot,
        pilot_case_library_pairs=len(pilot) * len(libraries),
        representative_cases=representatives,
        expected_case_library_pairs=len(representatives) * len(libraries),
        manifest_sha256=hashes, case_inventory_sha256=sha256(run / "plan.json"),
        no_schedule_changes=True, bulk_agent_dispatch=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "benchmarks/ascend")
    args = parser.parse_args()
    data = inventory()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "classic_inventory.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    )
    with (args.output_dir / "operator_categories.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data["operators"][0]))
        writer.writeheader()
        writer.writerows(data["operators"])
    print(json.dumps(dict(
        operators=len(data["operators"]), categories=len(data["categories"]),
        representatives=len(data["categories"]), cases=len(data["representative_cases"]),
        libraries=len(data["libraries"]), pairs=data["expected_case_library_pairs"],
        active_phase_pairs=data["pilot_case_library_pairs"],
    )))


if __name__ == "__main__":
    main()
