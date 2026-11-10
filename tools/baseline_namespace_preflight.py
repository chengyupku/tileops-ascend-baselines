#!/usr/bin/env python3
"""Check Native packet runtime literals without importing or running the packet.

This is a fast authoring check, not source approval or execution authorization.
Only the existing activation.B and $here/../runtime shell forms are understood.
Unknown forms are reported explicitly rather than guessed or executed.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re


def inspect(private):
    private = Path(private).resolve()
    findings, evidence = [], []

    def read(name):
        path = private / name
        raw = path.read_bytes()
        evidence.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()})
        return raw.decode()

    source = read("activation.py")
    tree = ast.parse(source)
    assignments = [n for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "B" for t in n.targets)]
    runtime = None
    if len(assignments) == 1:
        expr = assignments[0].value
        if (isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Div)
                and isinstance(expr.left, ast.Name) and expr.left.id == "D"
                and isinstance(expr.right, ast.Constant)
                and isinstance(expr.right.value, str)):
            runtime = expr.right.value
    if runtime is None or not re.fullmatch(r"runtime/[A-Za-z0-9_.-]+", runtime):
        findings.append({"kind": "unsupported_activation_runtime", "file": "activation.py"})
    for name in ("device.sh", "devicefs.sh"):
        shell = read(name)
        run_lines = [(i, line) for i, line in enumerate(shell.splitlines(), 1)
                     if re.match(r"\s*run=", line)]
        if len(run_lines) != 1:
            findings.append({"kind": "unsupported_shell_runtime", "file": name})
            continue
        line_no, line = run_lines[0]
        matches = re.findall(r"\$here/\.\./(runtime/[A-Za-z0-9_.-]+)", line)
        if len(matches) != 1:
            findings.append({"kind": "unsupported_shell_runtime", "file": name,
                             "line": line_no})
        elif runtime is not None and matches[0] != runtime:
            findings.append({"kind": "runtime_namespace_mismatch", "file": name,
                             "line": line_no, "declared": runtime, "actual": matches[0]})
    return {"schema": "native-runtime-literal-preflight-v1", "private": str(private),
            "status": "PASS" if not findings else "HOLD", "runtime": runtime,
            "findings": findings, "source_evidence": evidence,
            "packet_imported": False, "commands_executed": 0,
            "scientific_approval": False, "execution_authorization": False,
            "scope": "Runtime literal consistency only; does not establish full script correctness."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("private", type=Path)
    args = parser.parse_args()
    try:
        report = inspect(args.private)
    except (OSError, UnicodeError, SyntaxError) as exc:
        report = {"status": "HOLD", "error": str(exc), "commands_executed": 0}
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
