#!/usr/bin/env python3
"""Build an isolated snapshot with the library's original build script."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

WS = Path(__file__).resolve().parents[4]
BASE = WS / ".cache/all-pairs"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("library", choices=("ops-math",))
    parser.add_argument("--install-existing", action="store_true",
                        help="Install completed original CMake targets without repeating packaging.")
    parser.add_argument("--preflight", action="store_true",
                        help="CPU-only input integrity and missing-build-input report; never starts a build.")
    args = parser.parse_args()
    task = BASE / "remote_ops_math"
    snapshot = json.loads((task / "snapshot.json").read_text())
    source = task / "source"
    hashes_ok = all((source / n).is_file() and hashlib.sha256((source / n).read_bytes()).hexdigest() == h
                    for n, h in snapshot["sha256"].items())
    missing = [str(source / n) for n in ("build.sh", "CMakeLists.txt") if not (source / n).is_file()]
    result = dict(source_hashes_match=hashes_ok, source_file_count=len(snapshot["sha256"]),
                  missing_required_entry_files=missing, original_build_ready=hashes_ok and not missing,
                  complete_dependency_audit=False, device_commands=0,
                  required_local_files=str(task / "required_local_files.json"))
    if args.preflight:
        (task / "original_build_preflight.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        return 0 if result["original_build_ready"] else 2
    if not result["original_build_ready"]:
        raise RuntimeError("Original build input incomplete; run --preflight and supply required_local_files.json")
    lock = json.loads(Path("/run/lock/ascend-npu/device-0.lock").read_text())
    if lock.get("phase") != "running" or lock.get("child", {}).get("pid") != os.getpgrp() or lock.get("owner") != "allpairs-remote/ops-math":
        raise RuntimeError("The complete build requires the host npu-guard.")
    task = BASE / "remote_ops_math"
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    for key, leaf in (('TMPDIR','tmp'),('XDG_CACHE_HOME','xdg'),('ASCEND_CACHE_PATH','build-runtime-cache'),('TORCH_EXTENSIONS_DIR','torch_extensions'),('ASCEND_PROCESS_LOG_PATH','build-cann-logs')):
        directory = task / leaf
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(directory)
    snapshot = json.loads((task / "snapshot.json").read_text())
    source = task / "source"
    for name, expected in snapshot["sha256"].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Source snapshot changed: {name}")
    record = {
        "library": args.library, "ops": snapshot["ops"],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "snapshot": str(task / "snapshot.json"),
        "source_schedule_changes": False,
        "host_metadata_patches": snapshot.get("host_metadata_patches", []),
    }
    command = ["bash", "-c", "source " + shlex.quote(str(task / "remote_env.sh")) + "\nexec bash ./build.sh \"$@\"",
               "original-build", "--pkg", "--jit", "--soc=ascend950",
               "--ops=" + ",".join(snapshot["ops"]), "--vendor_name=allpairs_math",
               "-j8", "--build-type=Release"]
    # Transformer parses this as two arguments; Math/NN accept --ccache=off.
    command.extend(["--ccache", "off"] if args.library == "ops-transformer"
                   else ["--ccache=off"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for name in ("build.log", "install.log", "build_receipt.json"):
        previous = task / name
        if previous.exists():
            shutil.copy2(previous, task / f"{name}.{stamp}.prior")
    if args.install_existing:
        required = [source / "build/libcust_opapi.so", source / "build/libcust_opmaster_rt2.0.so"]
        if not all(p.is_file() and p.stat().st_size for p in required):
            raise RuntimeError("Original native libraries must already be fully linked.")
        record["build_mode"] = "install already linked original CMake targets; packaging was interrupted"
        record["reused_artifacts_sha256"] = {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in required}
        record["build_exit_code"] = None
        build_ready = True
    else:
        record["build_mode"] = "original --jit graph source package; requires graph JIT execution, not ACLNN fallback"
        record["build_command"] = command
        env = os.environ.copy()
        # CPack invokes a nested `cmake --build --target preinstall` without -j.
        # Keep that original build stage parallel as well.
        env["CMAKE_BUILD_PARALLEL_LEVEL"] = "8"
        with (task / "build.log").open("w") as log:
            result = subprocess.run(command, cwd=source, stdout=log, stderr=subprocess.STDOUT, env=env)
        record["build_exit_code"] = result.returncode
        build_ready = result.returncode == 0
    if build_ready:
        stage = task / "install"
        with (task / "install.log").open("w") as log:
            installed = subprocess.run(["cmake", "--install", str(source / "build"),
                                        "--prefix", str(stage)], stdout=log,
                                       stderr=subprocess.STDOUT)
        record["install_exit_code"] = installed.returncode
        package_vendors = stage / "packages/vendors"
        if installed.returncode == 0 and package_vendors.is_dir():
            final_vendors = stage / "vendors"
            final_vendors.mkdir(exist_ok=True)
            for vendor in package_vendors.iterdir():
                if vendor.is_dir():
                    shutil.copytree(vendor, final_vendors / vendor.name, symlinks=False,
                                    dirs_exist_ok=True)
                    selected = final_vendors / vendor.name
                    entry = selected / "bin/set_env.bash"
                    entry.parent.mkdir(parents=True, exist_ok=True)
                    entry.write_text(
                        "export ASCEND_CUSTOM_OPP_PATH=" + shlex.quote(str(selected)) +
                        ':${ASCEND_CUSTOM_OPP_PATH:-}\n' +
                        "export LD_LIBRARY_PATH=" + shlex.quote(str(selected / "op_api/lib")) +
                        ':${LD_LIBRARY_PATH:-}\n')
        record["api_libraries"] = [str(p) for p in stage.rglob("libcust_opapi.so")]
        record["device_configs"] = [str(p) for p in stage.rglob("config/ascend950/*.json")]
    record["finished_at"] = datetime.now(timezone.utc).isoformat()
    record["device_validated"] = False
    (task / "build_receipt.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)
    return record.get("install_exit_code", record["build_exit_code"])


if __name__ == "__main__":
    sys.exit(main())
