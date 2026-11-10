"""Validate remote benchmark evidence before deriving any public performance row.

This module uses only the standard library; it never imports an NPU runtime.
"""

import hashlib
import json
import math
import re
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path

PROTOCOL = "ascend950-profiler-locked-v1"
PUBLISHER = "openlibing-DevEnv-661495"
PUBLISHER_REPOSITORY = "https://github.com/yyttt6/TileOPs.github.io"
SHA = re.compile(r"[0-9a-f]{40}")
DENIED_TAGS = {"tileops", "reference", "torch-ref", "baseline", "eager"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def seal(snapshot):
    """Hash the exact evidence bundle, excluding its own inventory."""
    files = {
        str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(snapshot.rglob("*"))
        if p.is_file() and p.name != "sha256.json"
    }
    (snapshot / "sha256.json").write_text(json.dumps(files, indent=2) + "\n")


def validate_run(snapshot, name, verify_only):
    directory = snapshot / name
    receipt = read_json(directory / "receipt.json")
    result = read_json(directory / "result.json")
    require(receipt.get("status") == "passed" and receipt.get("guard_exit_code") == 0,
            f"{name}: unsuccessful guarded run")
    require(result.get("status") == "passed" and result.get("source_changed") is False
            and result.get("verify_only") is verify_only and result.get("failed") == 0
            and type(result.get("passed")) is int and result["passed"] > 0,
            f"{name}: incomplete or changed-source run")
    require(receipt.get("result") == result, f"{name}: receipt/result mismatch")
    environment = read_json(directory / "environment.json")
    require(environment.get("tilelang") == "0.1.15"
            and "Ascend950" in environment.get("device", "")
            and environment.get("verify_only") is verify_only,
            f"{name}: wrong runtime or device")
    rows = [json.loads(line) for line in (directory / "results.jsonl").read_text().splitlines()]
    require(len(rows) == result["passed"], f"{name}: row count mismatch")
    identities = set()
    for row in rows:
        identity = (row["op"], row["case_id"])
        require(identity not in identities, f"{name}: duplicate workload")
        identities.add(identity)
        require(row["status"] == ("verified" if verify_only else "measured"),
                f"{name}: unsuccessful workload")
        require("tileops" in row["implementations"], f"{name}: missing TileOPs")
        for impl in row["implementations"].values():
            check = impl["correctness"]
            require(type(check.get("total_outputs")) is int and check["total_outputs"] > 0
                    and check.get("checked_outputs") == check["total_outputs"]
                    and not check.get("unchecked_reason"), f"{name}: incomplete correctness")
    return rows, environment


def validate(snapshot):
    snapshot = Path(snapshot).resolve()
    if read_json(snapshot / "meta.json").get("schema_version") == 2:
        from nightly_snapshot import validate as validate_nightly
        return validate_nightly(snapshot)
    inventory = read_json(snapshot / "sha256.json")
    require(isinstance(inventory, dict) and bool(inventory), "Missing evidence inventory")
    actual = {str(p.relative_to(snapshot)) for p in snapshot.rglob("*")
              if p.is_file() and p.name != "sha256.json"}
    require(set(inventory) == actual, "Evidence file inventory mismatch")
    for name, digest in inventory.items():
        path = snapshot / name
        require(not path.is_symlink() and path.resolve().is_relative_to(snapshot),
                "Evidence path escapes snapshot")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == digest, f"Changed evidence: {name}")
    meta = read_json(snapshot / "meta.json")
    require(meta.get("schema_version") == 1 and meta.get("measurement_protocol") == PROTOCOL
            and meta.get("publisher") == PUBLISHER and meta.get("status") == "passed"
            and meta.get("publisher_repository") == PUBLISHER_REPOSITORY,
            "Not a remote official snapshot")
    require(bool(SHA.fullmatch(meta.get("commit", ""))), "Missing exact TileOPs commit")
    require(str(meta.get("run_id", "")).isdigit(), "Missing GitHub run identity")
    correct, _ = validate_run(snapshot, "correctness", True)
    correct_ids = {(r["op"], r["case_id"]): r for r in correct}
    require(meta.get("correctness_count") == len(correct), "Correctness coverage mismatch")
    if meta.get("comparison_count") == 0:
        require(not (snapshot / "performance").exists(), "Empty snapshot contains performance data")
        return meta, []
    rows, env = validate_run(snapshot, "performance", False)
    require(read_json(snapshot / "correctness/sources.json")
            == read_json(snapshot / "performance/sources.json"),
            "Correctness and performance used different sources")
    require(meta.get("clock_verified") is True and meta.get("guard_verified") is True,
            "Unverified clock or reservation")
    clock = read_json(snapshot / "performance/clock-readbacks.json")
    targets = env["clock_policy"]["component_mhz"]
    require(len(targets) == 2 and all(type(v) is int and v > 0 for v in targets),
            "Invalid clock targets")
    require(clock.get("before") and clock.get("after_restore") and clock.get("samples"),
            "Missing lock/restore readbacks")
    calls = sum(len(r["implementations"]) for r in rows)
    require(len(clock["samples"]) >= 1 + 33 * calls, "Missing per-call clock checks")
    for sample in clock["samples"]:
        require([v["mhz"] for v in sample] == targets, "Clock target mismatch")
    provenance = meta.get("baseline_provenance", {})
    require(bool(provenance), "Missing baseline provenance")
    for tag, source in provenance.items():
        require(tag not in DENIED_TAGS and bool(re.fullmatch(r"[a-z][a-z0-9-]*", tag)),
                "Invalid comparison implementation tag")
        require(bool(SHA.fullmatch(source.get("commit", "")))
                and source.get("repository", "").startswith("https://github.com/")
                and bool(re.fullmatch(r"[0-9a-f]{64}", source.get("source_sha256", ""))),
                f"Missing source identity: {tag}")
        require(inventory.get(source.get("evidence_path")) == source["source_sha256"],
                f"Missing baseline source evidence: {tag}")
    eligible = []
    for row in rows:
        require((row["op"], row["case_id"]) in correct_ids, "Workload missing correctness CI")
        reference_row = correct_ids[(row["op"], row["case_id"])]
        require(row.get("params") == reference_row.get("params")
                and row.get("seed") == reference_row.get("seed"),
                "Correctness/performance workload or seed mismatch")
        for tag, impl in row["implementations"].items():
            require(tag == "tileops" or tag in provenance, "Undeclared baseline")
            m = impl["measurement"]
            require(m.get("protocol") == PROTOCOL
                    and m.get("profiler") == "torch_npu.profiler.profile"
                    and m.get("metric") == "mean_device_busy_union_ms"
                    and (m.get("warmup"), m.get("repeats"), m.get("flush_bytes")) == (3, 15, 1 << 30),
                    "Measurement protocol mismatch")
            samples = m["samples_ms"]
            require(len(samples) == 15 and all(positive(v) for v in samples), "Invalid samples")
            require(positive(m["latency_ms"])
                    and math.isclose(m["latency_ms"], statistics.mean(samples), rel_tol=1e-10),
                    "Published mean disagrees with samples")
            require(len(m["records"]) == 15, "Missing profiler records")
            for sample, record in zip(samples, m["records"], strict=True):
                require(record["profile_excluded_kernel_count"] == 0
                        and record["profile_kernel_count"] > 0
                        and math.isclose(record["latency_ms"], sample, rel_tol=1e-10),
                        "Invalid or contaminated profiler record")
        if len(row["implementations"]) > 1:
            eligible.append(row)
    require(len(eligible) == meta.get("comparison_count") and bool(eligible),
            "Matched comparison count mismatch")
    return meta, eligible


def write_xml(rows, path):
    root = ET.Element("testsuites")
    suite = ET.SubElement(root, "testsuite", name="ascend950", tests=str(len(rows)))
    for row in rows:
        case = ET.SubElement(suite, "testcase", name=f"test_bench[{row['case_id']}]")
        props = ET.SubElement(case, "properties")

        def prop(name, value, parent=props):
            ET.SubElement(parent, "property", name=name, value=str(value))

        prop("op", row["op"])
        if row.get("op_module"):
            prop("op_module", row["op_module"])
        ours = row["implementations"]["tileops"]["measurement"]["latency_ms"]
        for tag, impl in row["implementations"].items():
            m = impl["measurement"]
            prop(f"{tag}_device_busy_ms", m["latency_ms"])
            prop(f"{tag}_timing", "torch_npu.profiler.profile")
            prop(f"{tag}_n_samples", 15)
            if tag != "tileops":
                prop(f"{tag}_ratio", m["latency_ms"] / ours)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
