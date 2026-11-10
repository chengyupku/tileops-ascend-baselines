"""Validate an optional, explicitly instrumented host build without editing vendor.

This records reuse of the reviewed build. It never labels the observed build as
pristine, and never upgrades path snapshots into consumed-byte attestation.
"""
import hashlib
import json
from pathlib import Path


def validate(build, vendor, cache, workspace):
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    descriptor = build / "observed_deployment.json"
    if not descriptor.is_file():
        return None
    row = json.loads(descriptor.read_text())
    if row.get("schema") != "deepgemm-observed-deployment/v1" or not row.get("diagnostic_only"):
        raise RuntimeError("Unknown observed deployment descriptor")
    source = Path(row["source_root"]).resolve()
    if not source.is_relative_to(cache.resolve()) or Path(row["package_root"]).resolve() != build:
        raise RuntimeError("Observed source/package must stay in the provider cache")
    actual = {str(p.relative_to(source)): sha(p) for p in source.rglob("*") if p.is_file()}
    if actual != row["observed_source_sha256"]:
        raise RuntimeError("Observed source inventory changed")
    pristine = row["original_vendor_sha256"]
    if any(not (vendor / p).is_file() or sha(vendor / p) != h for p, h in pristine.items()):
        raise RuntimeError("Original vendor differs from observed build baseline")
    changed = sorted(p for p, h in actual.items() if pristine.get(p) != h)
    expected = ["third-party/deep-jit/include/deep_jit/backend/ascend/driver.hpp",
                "third-party/deep-jit/include/deep_jit/utils/native_load_trace.hpp"]
    if changed != expected or row["allowed_observer_changes"] != expected:
        raise RuntimeError("Observed build changed more than the host loader observer")
    for key in ("build_receipt", "host_compile_input", "trace_dso", "observer_patch", "binary"):
        record = row[key]
        if not Path(record["path"]).is_file() or sha(record["path"]) != record["sha256"]:
            raise RuntimeError("Observed build evidence changed: " + key)
    receipt = json.loads(Path(row["build_receipt"]["path"]).read_text())
    if receipt.get("schema") != "nlt-diagnostic-host-build/v1" or not receipt.get("diagnostic_only"):
        raise RuntimeError("Missing genuine observed host build receipt")
    if receipt["binary"]["sha256"] != row["binary"]["sha256"]:
        raise RuntimeError("Reused host binary does not match actual build")
    if receipt["actual_compile_input"]["sha256"] != row["host_compile_input"]["sha256"]:
        raise RuntimeError("Actual host compile input differs")
    # The historical compiler receipt retains the original paths and hashes.
    historical = receipt["source_sha256"]
    prefix = next(k[:-len("README.md")] for k in historical if k.endswith("/source/README.md"))
    compile_sources = {p: h for p, h in actual.items() if p in ("README.md", "setup.py") or
                       (p.startswith(("csrc/", "deep_gemm/", "third-party/deep-jit/include/",
                                      "third-party/magic_enum/include/")) and
                        Path(p).suffix in (".py", ".hpp", ".h", ".cpp"))}
    if historical != {prefix + p: h for p, h in compile_sources.items()}:
        raise RuntimeError("Observed sources differ from genuine compiler receipt")
    hashes = {str((source / p).relative_to(workspace)): h for p, h in actual.items()}
    hashes[str(descriptor.relative_to(workspace))] = sha(descriptor)
    for key in ("build_receipt", "host_compile_input", "trace_dso", "observer_patch"):
        record = row[key]
        hashes[str(Path(record["path"]).relative_to(workspace))] = record["sha256"]
    return receipt, hashes
