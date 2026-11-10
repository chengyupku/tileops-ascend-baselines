#!/usr/bin/env python3
"""Stage a mechanical control-only fix for future local baseline namespaces.

The existing 32/28/4 GiB launcher enters its cgroup before resource admission,
so other waiting controllers see an unregistered active cgroup. Keep it empty
until the existing admission.reservation() performs entry and registration
under the host admission mutex. No kernel, schedule, memory limit, guard,
numerical checker, prior source manifest, or historical claim is changed.

Usage: python tools/baseline_resource_entry.py stage --source-dir OLD_CONTROL
       --output-dir NEW_CONTROL_PATCH
Copy the two emitted control files into the future private namespace BEFORE
generating its new source pins. Existing sealed or attempted packages stay intact.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import time


ENTRY = "(G / 'control/cgroup.procs').write_text(str(os.getpid()))"


def is_entry(node):
    return isinstance(node, ast.Expr) and ast.unparse(node) == ENTRY


def lock_context(node):
    return (isinstance(node, ast.With)
            and any("lockfile(LOCKS / 'admission.lock'" in ast.unparse(item.context_expr)
                    for item in node.items))


def repair(launch, admission):
    """Only the known launcher/admission control ABI is accepted; no guessing."""
    lt, at = ast.parse(launch), ast.parse(admission)
    entries = [n for n in ast.walk(lt) if is_entry(n)]
    if len(entries) != 1:
        raise ValueError("expected exactly one early launcher control-cgroup entry")
    reservation = next((n for n in at.body if isinstance(n, ast.FunctionDef) and n.name == "reservation"), None)
    if reservation is None:
        raise ValueError("existing reservation function missing")
    protected = []
    for mutex in ast.walk(reservation):
        if not lock_context(mutex):
            continue
        for branch in ast.walk(mutex):
            if not isinstance(branch, ast.If) or ast.unparse(branch.test) != "snap['admitted']":
                continue
            joins = [i for i, n in enumerate(branch.body) if is_entry(n)]
            leases = [i for i, n in enumerate(branch.body)
                      if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
                      and isinstance(n.value.func, ast.Name) and n.value.func.id == "Reservation"]
            if len(joins) == 1 and len(leases) == 1 and leases[0] == joins[0] + 1:
                protected.append((joins[0], leases[0]))
    if len(protected) != 1:
        raise ValueError("cannot prove existing admitted entry plus lifecycle registration under shared mutex")
    if not any(isinstance(n, ast.Call) and ast.unparse(n.func) == "fcntl.flock"
               and "LOCK_EX" in ast.unparse(n) for n in ast.walk(reservation)):
        raise ValueError("existing exclusive admission/slot locking missing")
    # Verify this really is the known bounded launcher, not a numerical module.
    if "32 << 30" not in ast.unparse(lt) or "('control', 4)" not in ast.unparse(lt) or "('payload', 28)" not in ast.unparse(lt):
        raise ValueError("unsupported launcher resource ABI; requires original 32/28/4 GiB")
    lines = launch.splitlines(keepends=True)
    entry = entries[0]
    indent = lines[entry.lineno - 1][:entry.col_offset]
    lines[entry.lineno - 1:entry.end_lineno] = [indent + "# Remain outside owner cgroup while waiting; reservation() enters and registers atomically.\n"]
    fixed = "".join(lines)
    # Compare ASTs after deleting this single operational statement.
    class RemoveEntry(ast.NodeTransformer):
        def visit_Expr(self, node):
            return None if is_entry(node) else self.generic_visit(node)
    expected = RemoveEntry().visit(ast.parse(launch))
    if ast.dump(ast.parse(fixed), include_attributes=False) != ast.dump(expected, include_attributes=False):
        raise ValueError("unexpected control delta")
    return fixed, admission


def stage(source, output):
    source, output = Path(source).resolve(), Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("output must be a fresh directory; never patch old sealed source")
    values = {}
    for name in ("launch.py", "admission.py"):
        path = source / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 512 * 1024:
            raise ValueError("missing or oversized regular control source: " + name)
        values[name] = path.read_text()
    fixed = repair(values["launch.py"], values["admission.py"])
    output.mkdir(parents=True)
    sha = lambda text: hashlib.sha256(text.encode()).hexdigest()
    files = []
    for name, body in zip(("launch.py", "admission.py"), fixed):
        (output / name).write_text(body)
        files.append({"name": name, "source": str(source / name), "source_sha256": sha(values[name]),
                      "output": str(output / name), "output_sha256": sha(body)})
    lifecycle = source / "reservation_lifecycle.py"
    if (lifecycle.is_file() and not (source / "metadata05.py").exists()
            and "from metadata05 import save" in lifecycle.read_text()):
        # Some Group B resource-only copies omitted this stdlib metadata writer.
        # Bind its existing bytes; do not invent another lifecycle implementation.
        helper = Path(__file__).resolve().parents[1] / ".cache/all-pairs/delivery_repair_20261023/pipeline_final/deployed/nn/metadata05.py"
        body = helper.read_text()
        expected = "b8bca60e0793e2abae63e64ba0f16aa8f6f9b128a9f477956b022619ba2b7303"
        if sha(body) != expected:
            raise ValueError("existing metadata-only lifecycle helper changed")
        (output / "metadata05.py").write_text(body)
        files.append({"name": "metadata05.py", "source": str(helper), "source_sha256": expected,
                      "output": str(output / "metadata05.py"), "output_sha256": expected})
    receipt = {"schema": "baseline-resource-entry-control-delta-v1", "at": time.time(), "files": files,
               "change": "remove only early launch control entry; retain admitted mutex-protected entry+registration",
               "limits_GiB": {"owner": 32, "payload": 28, "control": 4},
               "science_or_schedule_changed": False, "old_sources_modified": False,
               "scope": "future unattempted namespaces only; not baseline success or retry authorization"}
    (output / "RESOURCE_ENTRY_PATCH.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    command = sub.add_parser("stage")
    command.add_argument("--source-dir", required=True)
    command.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(stage(args.source_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
