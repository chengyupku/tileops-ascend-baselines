"""Run each previously untested exact GDN case once, sequentially."""
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
for batch in (64, 256, 512):
    case_id = "q35-gva-decode-b%d-float16" % batch
    out = HERE / case_id
    assert not (out / "guard.json").exists()
    command = [sys.executable, str(HERE / "run_guarded.py"), case_id, "180",
               str(HERE.parents[1] / ".tools/baselines/baseline-run"),
               "flash-linear-attention-npu", "python", str(HERE / "run_fla_gdn.py"), case_id]
    print("START " + case_id, flush=True)
    process = subprocess.run(command)
    row = {"key": ["flash-linear-attention-npu", "GDNFwdOp", case_id],
           "verified": False, "guard_rc": process.returncode,
           "checker": str(out / "checker.json"), "guard": str(out / "guard.json")}
    if (out / "checker.json").exists():
        checker = json.loads((out / "checker.json").read_text())
        row.update(checker_status=checker["status"], stage=checker.get("stage"),
                   error=checker.get("error"), verification=checker.get("verification"))
        row["verified"] = checker["status"] == "verified"
    row["status"] = "verified_pending_registration" if row["verified"] else "original_check_not_passed"
    (out / "RESULT.json").write_text(json.dumps(row, indent=2) + "\n")
    with (HERE / "RESULTS.jsonl").open("a") as stream:
        stream.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)
