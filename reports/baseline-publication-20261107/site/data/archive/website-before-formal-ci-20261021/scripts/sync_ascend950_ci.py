"""Fetch a successful Ascend 950 CI artifact, validate it, then atomically publish JSON."""

import argparse
import hashlib
import io
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from ascend950_data import require, validate_ci

REPOSITORY = "yyttt6/TileOPs"
API = f"https://api.github.com/repos/{REPOSITORY}"


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        result = super().redirect_request(req, fp, code, msg, headers, newurl)
        if result and urllib.parse.urlsplit(req.full_url).netloc != urllib.parse.urlsplit(newurl).netloc:
            result.remove_header("Authorization")
        return result


def fetch(url, *, binary=False):
    require(url.startswith("https://api.github.com/"), "Unexpected API host")
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "TileOPs-950-docs"}
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    opener = urllib.request.build_opener(SafeRedirect())
    for attempt in range(3):
        try:
            with opener.open(urllib.request.Request(url, headers=headers), timeout=60) as response:
                body = response.read(64 * 1024 * 1024 + 1)
                require(len(body) <= 64 * 1024 * 1024, "Artifact exceeds size limit")
            return body if binary else json.loads(body)
        except (urllib.error.URLError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(2)


def from_evidence(run, receipt, environment, rows, hashes):
    require(receipt["status"] == "passed", "Failed benchmark receipt")
    summary = receipt["result"]
    # Keep public version evidence; do not publish host PIDs, owners or local paths.
    env = {k: environment[k] for k in ("device", "tilelang", "torch", "torch_npu", "cann_components", "python", "ops")}
    return validate_ci({
        "schema": "ascend950-ci/v1",
        "run": {"id": run["id"], "commit": run["head_sha"], "branch": run["head_branch"],
                "url": run["html_url"], "conclusion": run["conclusion"], "completed_at": receipt["finished_at"]},
        "environment": env, "guard_exit_code": receipt["guard_exit_code"],
        "source_changed": summary["source_changed"], "verify_only": summary["verify_only"],
        "passed": summary["passed"], "failed": summary["failed"], "source_sha256": hashes,
        "cases": rows,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/ascend950/current-ci.json"))
    args = parser.parse_args()
    runs = fetch(f"{API}/actions/workflows/ascend950.yml/runs?branch=ascend950&status=success&per_page=20")["workflow_runs"]
    runs = [r for r in runs if r["head_branch"] == "ascend950" and r["event"] in ("push", "workflow_dispatch")
            and r["head_repository"]["full_name"] == REPOSITORY]
    require(bool(runs), "No successful trusted Ascend 950 CI run")
    run = runs[0]
    artifacts = fetch(f"{API}/actions/runs/{run['id']}/artifacts?per_page=100")["artifacts"]
    artifacts = [a for a in artifacts if a["name"] == "ascend950-device" and not a["expired"]]
    require(len(artifacts) == 1, "Missing or ambiguous CI artifact; preserve the live site")
    raw = fetch(artifacts[0]["archive_download_url"], binary=True)
    evidence = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for name in ("receipt.json", "environment.json", "results.jsonl"):
            names = [n for n in archive.namelist() if n == f"manifest-device/{name}" or n.endswith(f"/manifest-device/{name}")]
            require(len(names) == 1, f"Missing/ambiguous artifact member: {name}")
            require(archive.getinfo(names[0]).file_size < 16 * 1024 * 1024, "Oversized artifact member")
            evidence[name] = archive.read(names[0])
    result = from_evidence(run, json.loads(evidence["receipt.json"]), json.loads(evidence["environment.json"]),
                           [json.loads(line) for line in evidence["results.jsonl"].splitlines() if line],
                           {n: hashlib.sha256(v).hexdigest() for n, v in evidence.items()})
    if args.output.exists():
        previous = validate_ci(json.loads(args.output.read_text()))
        require(result["run"]["completed_at"] >= previous["run"]["completed_at"], "Refusing a stale CI snapshot")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(args.output)
    print(f"CI {run['id']}: {result['passed']} verified cases at {run['head_sha']}")


if __name__ == "__main__":
    main()
