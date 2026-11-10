"""Publication rejects incomplete, contaminated and numerically inconsistent receipts."""

import json

import pytest
from ascend950_snapshot import seal
from nightly_snapshot import COVERAGE, PROTOCOL, build_cache, validate


def bundle(tmp_path):
    check = dict(checked_outputs=1, total_outputs=1, unchecked_reason=None)
    m = dict(
        warmup=3,
        repeats=15,
        flush_bytes=1 << 30,
        measurement_protocol=PROTOCOL,
        profile_coverage_protocol=COVERAGE,
        latency_ms=2.0,
        latency_samples_ms=[2.0] * 15,
        profile_excluded_kernel_count=0,
    )
    row = dict(
        op="ReluFwdOp",
        case_id="x",
        params={"shape": [4]},
        status="passed",
        input_preserved=True,
        raw_replayed=True,
        correctness=check,
        measurement=m,
        coverage=[
            dict(
                complete_sha256="a" * 64,
                result=dict(latency_ms=2.0, profile_excluded_kernel_count=0),
            )
            for _ in range(15)
        ],
    )
    base = dict(
        rows=[
            dict(
                library="native",
                op="ReluFwdOp",
                case="x",
                workload=dict(params={"shape": [4]}),
                latency_us=3000.0,
            )
        ],
        tileops_rows=[],
    )
    meta = dict(
        schema_version=2,
        publisher="server2-physical32",
        publisher_repository="https://github.com/yyttt6/TileOPs.github.io",
        status="passed",
        measurement_protocol=PROTOCOL,
        commit="b" * 40,
        date="2026-11-03",
        operators=["ReluFwdOp"],
        case_ids={"ReluFwdOp": ["x"]},
        fresh_cases=1,
    )
    receipts = {
        "ReluFwdOp": dict(
            guard_exit_code=0,
            result=dict(status="passed", failed=0, passed=1, source_changed=False),
            hardware=dict(cube=32, vector=64, clock_policy="platform-default"),
        )
    }
    values = {
        "meta": meta,
        "measurements": [row],
        "baseline-before": base,
        "receipts": receipts,
        "baseline-cache": build_cache(base, [row], meta["commit"], meta["date"]),
    }
    for name, value in values.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(value))
    seal(tmp_path)
    return tmp_path


def test_roundtrip_preserves_baseline(tmp_path):
    root = bundle(tmp_path)
    validate(root)
    data = json.loads((root / "baseline-cache.json").read_text())
    assert data["rows"][0]["latency_us"] == 3000.0
    assert data["rows"][0]["ratio"] == 1.5


@pytest.mark.parametrize(
    "failure", ["guard", "samples", "mean", "coverage", "contract", "native", "source"]
)
def test_reject_bad_evidence(tmp_path, failure):
    root = bundle(tmp_path)
    name = "measurements"
    rows = json.loads((root / "measurements.json").read_text())
    if failure == "samples":
        rows[0]["measurement"]["latency_samples_ms"].pop()
    if failure == "mean":
        rows[0]["measurement"]["latency_ms"] = 3
    if failure == "coverage":
        rows[0]["coverage"][-1]["result"]["latency_ms"] = 3
    if failure == "contract":
        rows[0]["params"] = {"shape": [8]}
    if failure in ("guard", "source"):
        name = "receipts"
        rows = json.loads((root / "receipts.json").read_text())
        if failure == "guard":
            rows["ReluFwdOp"]["guard_exit_code"] = 76
        else:
            rows["ReluFwdOp"]["result"]["source_changed"] = True
    if failure == "native":
        name = "baseline-cache"
        rows = json.loads((root / "baseline-cache.json").read_text())
        rows["rows"][0]["latency_us"] = 1
    (root / f"{name}.json").write_text(json.dumps(rows))
    seal(root)
    with pytest.raises(AssertionError):
        validate(root)


@pytest.mark.parametrize("order", [float("inf"), float("-inf")])
def test_vector_norm_infinity_cache_contract(tmp_path, order):
    root = bundle(tmp_path)
    base = json.loads((root / "baseline-before.json").read_text())
    measured = json.loads((root / "measurements.json").read_text())
    base["rows"][0]["op"] = measured[0]["op"] = "VectorNormFwdOp"
    params = {"shape": [4], "dim": -1, "ord": order}
    measured[0]["params"] = params
    base["rows"][0]["workload"]["params"] = dict(
        params, ord="Infinity" if order > 0 else "-Infinity"
    )
    before = json.dumps(base, sort_keys=True)
    result = build_cache(base, measured, "b" * 40, "2026-11-05")
    assert result["rows"][0]["ratio"] == 1.5
    assert json.dumps(base, sort_keys=True) == before
    assert result["rows"][0]["workload"] == base["rows"][0]["workload"]
    for changed in [dict(params, ord=-order), dict(params, ord=2), dict(params, dim=0)]:
        measured[0]["params"] = changed
        with pytest.raises(AssertionError, match="Changed cached contract"):
            build_cache(base, measured, "b" * 40, "2026-11-05")


def test_nightly_includes_named_ascend_variants():
    from nightly_device import registered_ascend_ops

    registered = {
        "BatchNormFwdOp": {"ascend_train": object(), "ascend_infer": object()},
        "InstanceNormFwdOp": {"ascend_train": object()},
        "ReluFwdOp": {"ascend": object()},
        "OtherOp": {"cuda": object(), "not_ascend": object()},
    }
    assert registered_ascend_ops(registered) == [
        "BatchNormFwdOp", "InstanceNormFwdOp", "ReluFwdOp"
    ]
