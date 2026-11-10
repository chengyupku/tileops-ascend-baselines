"""Publication gates for the 950 evidence, independent of device execution."""

import copy
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest
from ascend950_data import validate_ci, validate_performance
from sync_ascend950_ci import SafeRedirect

ROOT = Path(__file__).resolve().parents[1]


def dataset(name):
    return json.loads((ROOT / "data/ascend950" / name).read_text())


def test_committed_evidence_is_valid():
    validate_ci(dataset("current-ci.json"))
    validate_performance(dataset("development-v2.json"))


@pytest.mark.parametrize("field,value", [("guard_exit_code", 76), ("source_changed", True), ("verify_only", False)])
def test_ci_rejects_invalid_receipts(field, value):
    data = dataset("current-ci.json")
    data[field] = value
    with pytest.raises(ValueError):
        validate_ci(data)


def test_ci_rejects_missing_output_checks_and_duplicate_cases():
    data = dataset("current-ci.json")
    duplicate = copy.deepcopy(data)
    duplicate["cases"][1] = duplicate["cases"][0]
    with pytest.raises(ValueError):
        validate_ci(duplicate)
    data["cases"][0]["implementations"]["tileops"]["correctness"]["checked_outputs"] = 0
    with pytest.raises(ValueError):
        validate_ci(data)


def test_performance_cannot_be_relabelled_as_locked_or_have_samples_changed():
    data = dataset("development-v2.json")
    locked = copy.deepcopy(data)
    locked["clock_locked"] = True
    with pytest.raises(ValueError):
        validate_performance(locked)
    data["cases"][0]["tileops"]["samples_ms"][0] *= 2
    with pytest.raises(ValueError):
        validate_performance(data)


def test_artifact_redirect_does_not_forward_credentials():
    request = urllib.request.Request("https://api.github.com/artifact", headers={"Authorization": "Bearer test-only"})
    redirected = SafeRedirect().redirect_request(request, None, 302, "Found", {}, "https://storage.example/artifact.zip")
    assert redirected.get_header("Authorization") is None


def test_rendered_pages_keep_every_case_and_distinguish_protocols(tmp_path):
    subprocess.run([sys.executable, "scripts/gen_ascend950_pages.py", "--out-dir", str(tmp_path)], cwd=ROOT, check=True)
    data = dataset("development-v2.json")
    for suffix in (".md", ".zh.md"):
        performance = (tmp_path / ("performance" + suffix)).read_text()
        assert performance.count("<div class=\"datatable\" markdown=\"1\">") == len({r["op"] for r in data["cases"]})
        assert sum(line.startswith("| `") for line in performance.splitlines()) == len(data["cases"])
        assert "0.1.15" in performance and "v2" in performance
        assert "Ascend 950" in (tmp_path / ("index" + suffix)).read_text()
    assert (tmp_path / "data/development-v2.json").read_bytes() == (ROOT / "data/ascend950/development-v2.json").read_bytes()
