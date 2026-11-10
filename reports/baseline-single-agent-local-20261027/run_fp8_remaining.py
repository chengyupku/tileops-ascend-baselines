"""Sequential foreground cases, each with its own complete host reservation."""
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
scope = json.loads((HERE / "SCOPE.json").read_text())
keys = [r["key"] for r in scope if r["key"][:2] == ["ops-transformer", "GemmFP8FwdOp"]]
keys = sorted(keys, key=lambda k: ("m128" not in k[2], k[2]))
first = "ds-v3-o-proj-m128-s1-float8_e4m3fn-bfloat16"
for i, key in enumerate(keys):
    if key[2] == first:
        continue
    attempt = "ops-transformer-fp8-remaining-%02d" % i
    out = HERE / attempt
    if (out / "guard.json").exists():
        raise RuntimeError("Refuse automatic replay of existing attempt: " + attempt)
    cmd = [sys.executable, str(HERE / "run_guarded.py"), attempt, "240", "bash",
           str(HERE / "ops-transformer-python"), str(HERE / "run_fp8_padded.py"),
           attempt, key[2]]
    print("START " + key[2], flush=True)
    p = subprocess.run(cmd)
    row = {"key": key, "attempt": attempt, "guard_rc": p.returncode,
           "checker": str(out / "checker.json"), "guard": str(out / "guard.json"),
           "verified": False, "registered": False, "performance_measured": False}
    if (out / "checker.json").exists():
        checker = json.loads((out / "checker.json").read_text())
        row.update(checker_status=checker["status"], verification=checker.get("verification"),
                   error=checker.get("error"), inputs_preserved=checker.get("inputs_preserved"))
        row["verified"] = checker["status"] == "verified"
    row["status"] = ("full_checker_verified_pending_registration" if row["verified"]
                     else "attempt_incomplete_or_failed")
    (out / "RESULT.json").write_text(json.dumps(row, indent=2) + "\n")
    with (HERE / "RESULTS.jsonl").open("a") as stream:
        stream.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)
