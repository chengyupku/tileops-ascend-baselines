"""Synthetic publication fixtures; these tests make no device-performance claim."""

import copy
import hashlib
import json
import os
import shutil
from pathlib import Path

import ascend950_snapshot as snapshot
import gen_bench_pages as pages
import pytest
import yaml
from remote_preflight import inspect
from render_official_ascend950 import render


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "snapshot"
    root.mkdir()
    baseline = root / "baseline-sources/native.py"
    baseline.parent.mkdir()
    baseline.write_text("# Synthetic test-only provenance\n")
    digest = hashlib.sha256(baseline.read_bytes()).hexdigest()
    meta = {
        "schema_version": 1, "status": "passed", "measurement_protocol": snapshot.PROTOCOL,
        "publisher": snapshot.PUBLISHER, "publisher_repository": snapshot.PUBLISHER_REPOSITORY,
        "commit": "a" * 40, "run_id": "123", "run_attempt": "1", "date": "2026-10-21",
        "gpu": "Ascend950DT", "correctness_count": 1, "comparison_count": 1,
        "clock_verified": True, "guard_verified": True,
        "baseline_provenance": {"torch": {
            "repository": "https://github.com/Ascend/pytorch", "commit": "b" * 40,
            "source_sha256": digest, "evidence_path": "baseline-sources/native.py",
        }},
    }
    write(root / "meta.json", meta)
    check = {"checked_outputs": 1, "total_outputs": 1, "unchecked_reason": None}
    measurement = {
        "protocol": snapshot.PROTOCOL, "profiler": "torch_npu.profiler.profile",
        "metric": "mean_device_busy_union_ms", "warmup": 3, "repeats": 15,
        "flush_bytes": 1 << 30, "latency_ms": 0.5, "samples_ms": [0.5] * 15,
        "records": [{"latency_ms": 0.5, "profile_excluded_kernel_count": 0,
                     "profile_kernel_count": 1}] * 15,
    }
    for kind, verification in (("correctness", True), ("performance", False)):
        result = {"status": "passed", "source_changed": False, "verify_only": verification,
                  "passed": 1, "failed": 0}
        write(root / kind / "result.json", result)
        write(root / kind / "receipt.json", {"status": "passed", "guard_exit_code": 0,
                                             "result": result})
        write(root / kind / "environment.json", {"tilelang": "0.1.15", "device": "Ascend950DT",
              "verify_only": verification, "clock_policy": {"component_mhz": [1500, 1500]}})
        write(root / kind / "sources.json", {"source.py": "c" * 64})
        impl = {"correctness": check}
        if not verification:
            impl["measurement"] = measurement
        row = {"op": "AbsFwdOp", "case_id": "16M-float16", "seed": 1,
               "params": {"dtype": "float16"}, "status": "verified" if verification else "measured",
               "implementations": {"tileops": impl}}
        if not verification:
            row["implementations"]["torch"] = copy.deepcopy(impl)
        (root / kind / "results.jsonl").write_text(json.dumps(row) + "\n")
    write(root / "performance/clock-readbacks.json", {
        "before": [{"mhz": 100}, {"mhz": 100}],
        "after_restore": [{"mhz": 100}, {"mhz": 100}],
        "samples": [[{"mhz": 1500}, {"mhz": 1500}]] * 67,
    })
    snapshot.seal(root)
    return root


def test_validated_rows_derive_upstream_xml(bundle, tmp_path):
    _, rows = snapshot.validate(bundle)
    xml = tmp_path / "bench.xml"
    snapshot.write_xml(rows, xml)
    parsed, failures, skips = pages.parse_bench_xml(str(xml))
    assert not failures and not skips
    assert parsed[0]["impls"]["torch"]["ratio"] == 1
    assert parsed[0]["impls"]["tileops"]["device_busy_ms"] == 0.5


@pytest.mark.parametrize("change", [
    "wrong-publisher", "changed-source", "unchecked-output", "wrong-seed",
    "unlocked", "short-run", "wrong-mean", "flush-contamination", "missing-baseline-source",
])
def test_rejects_ineligible_evidence(bundle, change):
    meta = snapshot.read_json(bundle / "meta.json")
    row_path = bundle / "performance/results.jsonl"
    row = json.loads(row_path.read_text())
    measurement = row["implementations"]["tileops"]["measurement"]
    if change == "wrong-publisher":
        meta["publisher"] = "local-development"
    elif change == "changed-source":
        write(bundle / "performance/sources.json", {"source.py": "d" * 64})
    elif change == "unchecked-output":
        row["implementations"]["tileops"]["correctness"]["checked_outputs"] = 0
    elif change == "wrong-seed":
        row["seed"] = 2
    elif change == "unlocked":
        meta["clock_verified"] = False
    elif change == "short-run":
        measurement["samples_ms"].pop()
    elif change == "wrong-mean":
        measurement["latency_ms"] = 0.1
    elif change == "flush-contamination":
        measurement["records"][0]["profile_excluded_kernel_count"] = 1
    elif change == "missing-baseline-source":
        meta["baseline_provenance"]["torch"]["source_sha256"] = "f" * 64
    write(bundle / "meta.json", meta)
    row_path.write_text(json.dumps(row) + "\n")
    snapshot.seal(bundle)
    with pytest.raises(ValueError):
        snapshot.validate(bundle)


def test_hash_mismatch_does_not_erase_previous_pages(bundle, tmp_path):
    out = tmp_path / "pages"
    out.mkdir()
    (out / "index.md").write_text("previous valid pages")
    (bundle / "meta.json").write_text("{}")
    with pytest.raises(ValueError, match="Changed evidence"):
        render(bundle, out)
    assert (out / "index.md").read_text() == "previous valid pages"


def test_empty_section_has_only_upstream_sections(tmp_path):
    render(None, tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["index.md"]
    text = (tmp_path / "index.md").read_text()
    assert [line for line in text.splitlines() if line.startswith("## ")] == [
        "## Environment", "## Method", "## Coverage", "## Data",
    ]
    assert "correctness.md" not in text and "archive" not in text


def test_ascend_method_matches_actual_shared_timer():
    text = "\n".join(pages.method_block({"measurement_protocol": snapshot.PROTOCOL}))
    assert "torch_npu.profiler.profile" in text and "15 measured" in text
    assert "arithmetic mean" in text and "1 GiB" in text
    assert "CUPTI" not in text and "median" not in text


def test_populated_snapshot_renders_upstream_family_page(bundle, tmp_path):
    repository = Path(__file__).resolve().parents[1]
    source = Path(os.environ.get("TILEOPS", repository / "TileOPs"))
    shutil.copytree(source / "src/tileops/manifest", bundle / "source/src/tileops/manifest",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    snapshot.seal(bundle)
    out = tmp_path / "pages"
    render(bundle, out)
    index = (out / "index.md").read_text()
    family = (out / "elementwise.md").read_text()
    assert "**NPU** Ascend950DT" in index
    assert "torch_npu.profiler.profile" in index and "CUPTI" not in index
    assert "https://github.com/yyttt6/TileOPs/commit/" in index
    assert "https://github.com/yyttt6/TileOPs.github.io/actions/runs/123" in index
    assert "AbsFwd" in family and "<table" in family and "<td>0.5000</td>" in family
    assert sorted(p.name for p in out.iterdir()) == ["elementwise.md", "index.md", "reading.md"]


def test_nightly_server2_at_shanghai_0300():
    root = Path(__file__).resolve().parents[1]
    workflow = yaml.safe_load((root / ".github/workflows/ascend950-nightly.yml").read_text())
    triggers = workflow.get("on", workflow.get(True))
    assert triggers["schedule"] == [{"cron": "0 19 * * *"}]
    assert "ascend950-production" in workflow["jobs"]["benchmark"]["runs-on"]
    assert workflow["jobs"]["publish"]["needs"] == "benchmark"
    assert "needs.contracts.outputs.source_commit" in json.dumps(workflow)


def test_unprovisioned_remote_fails_before_runtime_imports(tmp_path):
    config = tmp_path / "site.json"
    config.write_text('{"host_reservation_confirmed": false}')
    report = inspect(config)
    assert report["status"] == "blocked"
    assert report["hardware_started"] is False
    assert any("not confirmed" in error for error in report["errors"])
    assert any("clock_config" in error for error in report["errors"])
