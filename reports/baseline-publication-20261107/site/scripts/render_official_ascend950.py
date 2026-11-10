"""Render only remote official 950 snapshots with the upstream benchmark layout."""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from ascend950_snapshot import validate, write_xml

ROOT = Path(__file__).resolve().parents[1]


def build(snapshot, out):
    meta, rows = validate(snapshot) if snapshot is not None else ({}, [])
    if not rows:
        (out / "index.md").write_text(
            "# Benchmarks\n\n"
            "No eligible Ascend 950 performance snapshot has been published yet.\n\n"
            "## Environment\n\nAscend 950DT · TileLang 0.1.15.\n\n"
            "## Method\n\nOnly the dedicated remote nightly CI may publish results. "
            "Matched baselines, correctness, verified clock control and the shared profiler protocol "
            "must all pass before a workload is shown.\n\n"
            "## Coverage\n\n0 operators, 0 measured workloads.\n\n"
            "## Data\n\n| Page | Ops | Workloads |\n| --- | --- | --- |\n"
        )
        return
    xml = out / "bench_results.xml"
    write_xml(rows, xml)
    command = [sys.executable, str(ROOT / "scripts/gen_bench_pages.py"),
               "--bench-xml", str(xml),
               "--test-xml", str(xml),
               "--meta", str(snapshot / "meta.json"),
               "--commit", meta["commit"], "--date", meta["date"], "--gpu", meta["gpu"],
               "--tileops", str(snapshot / "source"),
               "--manifest-dir", str(snapshot / "source/src/tileops/manifest"),
               "--out-dir", str(out)]
    subprocess.run(command, cwd=ROOT, check=True)
    xml.unlink()


def render(snapshot, out):
    # Validate and render first: a rejected snapshot must preserve existing pages.
    with tempfile.TemporaryDirectory(prefix="ascend950-pages-") as directory:
        staging = Path(directory)
        build(snapshot, staging)
        out.mkdir(parents=True, exist_ok=True)
        for path in out.glob("*.md"):
            path.unlink()
        for path in staging.glob("*.md"):
            shutil.copy2(path, out / path.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "docs/benchmarks")
    args = parser.parse_args()
    render(args.snapshot, args.out_dir)


if __name__ == "__main__":
    main()
