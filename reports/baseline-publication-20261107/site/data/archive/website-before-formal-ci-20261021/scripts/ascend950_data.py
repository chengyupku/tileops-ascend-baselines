"""CPU-only validation of published Ascend 950 evidence."""

import math

LEGACY_PROTOCOL = "ascend950-isolated-ffts-v2"
FORMAL_PROTOCOL = "ascend950-profiler-locked-v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def key(row):
    return row["op"], row["case_id"]


def validate_ci(data):
    require(data["schema"] == "ascend950-ci/v1", "Unknown CI schema")
    require(data["run"]["conclusion"] == "success", "CI did not succeed")
    require(data["run"]["branch"] == "ascend950", "Wrong CI branch")
    require(data["guard_exit_code"] == 0 and data["source_changed"] is False,
            "CI reservation or source integrity failed")
    require(data["verify_only"] is True, "Expected a correctness-only CI snapshot")
    env = data["environment"]
    require("950" in env["device"] and env["tilelang"] == "0.1.15", "Wrong CI runtime")
    rows = data["cases"]
    require(rows and len({key(r) for r in rows}) == len(rows), "Missing/duplicate CI cases")
    require(set(env["ops"]) == {r["op"] for r in rows}, "Incomplete operator coverage")
    require(data["passed"] == len(rows) and data["failed"] == 0, "Incomplete CI results")
    for row in rows:
        require(row["status"] == "verified", f"Unverified case: {key(row)}")
        checks = row["implementations"]["tileops"]["correctness"]
        require(checks["checked_outputs"] == checks["total_outputs"] > 0,
                f"Unchecked outputs: {key(row)}")
        require(not checks["unchecked_reason"], f"Incomplete check: {key(row)}")
    return data


def validate_performance(data):
    require(data["schema"] == "ascend950-development/v1", "Unknown performance schema")
    require(data["protocol"] == LEGACY_PROTOCOL, "Wrong historical protocol")
    require(data["clock_locked"] is False and data["release_score_eligible"] is False,
            "Historical results must not masquerade as locked release scores")
    rows = data["cases"]
    require(rows and len({key(r) for r in rows}) == len(rows), "Missing/duplicate performance cases")
    for row in rows:
        for name in ("tileops", "baseline"):
            sample = row[name]
            values = sample["samples_ms"]
            require(sample["warmup"] == 3 and sample["repeats"] == len(values) == 15,
                    f"Wrong sampling protocol: {key(row)} / {name}")
            require(all(type(v) in (float, int) and math.isfinite(v) and v > 0 for v in values),
                    "Invalid sample")
            require(math.isclose(sum(values) / 15, sample["latency_ms"], rel_tol=1e-10),
                    "Reported latency differs from raw samples")
        ratio = row["baseline"]["latency_ms"] / row["tileops"]["latency_ms"]
        require(math.isclose(row["ratio"], ratio, rel_tol=1e-10), "Invalid ratio")
        require(row["passes_80pct"] == (ratio >= 0.8), "Invalid development threshold")
    return data
