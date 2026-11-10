"""Compile vendor kernels without importing torch, torch_npu or ACL.

Run with /opt/tileops-ci/runtime/bin/python. All compiler outputs, logs and the
receipt stay in .cache/all-pairs/flashmla. This never loads the resulting DSO.
The original setup.py imports torch_npu; it is inspected but deliberately not run.
"""
import ast
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import sysconfig

from native import CACHE, RECEIPT, VENDOR, WORKSPACE, hashes, source_files


def prepare_kernel_body(path):
    """Hide unchanged device definitions from CANN9.1's host pass."""
    source = path.read_text()
    markers = ('#include "config.h"', 'namespace ascend::prefill::sparse_fwd {',
               'inline uint32_t min(uint32_t x, uint32_t y) {')
    if any(source.count(marker) != 1 for marker in markers):
        raise RuntimeError("Sparse kernel host/device boundary changed; review compatibility staging")
    source = source.replace(markers[0], markers[0] + '\n#include "'
                            + str(Path(__file__).parent / "cann91_api.h")
                            + '"\n#if defined(__NPU_DEVICE__)\n', 1)
    source = source.replace(markers[1], '#endif\n' + markers[1] + '\n#if defined(__NPU_DEVICE__)\n', 1)
    source = source.replace(markers[2], '#else\ntemplate<typename Kernel>\n'
                            '__global__ __mix__(1, 2) void sparse_attn_fwd_kernel(const typename Kernel::Params params, const typename Kernel::AuxParams aux_params) {}\n'
                            '#endif\n\n' + markers[2], 1)
    path.write_text(source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-packages", type=Path, default=Path(sysconfig.get_path("purelib")))
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(CACHE)
    build_dir = CACHE / "builds" / datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    build_dir.mkdir(parents=True)
    receipt = {"schema": "flashmla-native-build/v1", "architecture": platform.machine(),
               "python": sys.version, "device_executed": False,
               "source_sha256": hashes(source_files()), "commands": [], "status": "building"}

    def save():
        receipt["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n")
        (build_dir / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")

    def fail(reason):
        receipt.update(status="deployment_blocked", reason=reason)
        save()
        print(reason)
        return 1

    save()
    if platform.machine() != "aarch64":
        return fail("This build path records only a native AArch64 deployment")
    toolkit = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann-9.1.0")).resolve()
    compiler = shutil.which("bisheng")
    purelib = args.site_packages.resolve()
    torch_root, npu_root = purelib / "torch", purelib / "torch_npu"
    includes = [Path(sysconfig.get_path("include")), torch_root / "include",
                torch_root / "include/torch/csrc/api/include", npu_root / "include",
                VENDOR / "csrc", VENDOR / "csrc/3rdparty/kerutils/include",
                toolkit / "aarch64-linux/include", toolkit / "aarch64-linux/asc/include"]
    runtime = toolkit / "aarch64-linux/lib64/libascendc_runtime.a"
    required = [*includes, runtime, npu_root / "lib/libtorch_npu.so", torch_root / "lib/libtorch.so"]
    missing = [str(p) for p in required if not p.exists()]
    if not compiler or missing:
        return fail(f"Missing compiler/headers/libraries: compiler={compiler}, paths={missing}")
    receipt.update(compiler=compiler, toolkit=str(toolkit), torch_root=str(torch_root), npu_root=str(npu_root))
    # The installed AArch64 libtorch exposes std::__cxx11 symbols (ABI=1).
    symbols = subprocess.run(["nm", "-D", str(torch_root / "lib/libc10.so")], capture_output=True, text=True, check=True).stdout
    if "__cxx11" not in symbols:
        return fail("Cannot establish the installed libtorch C++11 ABI from libc10 symbols")
    receipt["abi_evidence"] = "nm -D libc10.so contains __cxx11; _GLIBCXX_USE_CXX11_ABI=1"
    # Deployment-only compatibility: GCC11 lacks constexpr std::string. This
    # host diagnostic formatter is never used in device math or scheduling.
    staged = build_dir / "source"
    for original in source_files():
        target = staged / original.relative_to(VENDOR)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(original.read_bytes())
    common = staged / "csrc/api/common.h"
    before = "static constexpr std::string get_dynamic_enum_name(T value)"
    after = "static inline std::string get_dynamic_enum_name(T value)"
    if common.read_text().count(before) != 1:
        return fail("Host diagnostic compatibility patch did not match exactly once")
    common.write_text(common.read_text().replace(before, after))
    receipt["compatibility_edits"] = [{"file": "csrc/api/common.h", "before": before, "after": after,
                                       "reason": "GCC11 std::string is not a C++20 literal type; runtime error-message formatter only"}]
    prepare_kernel_body(staged / "csrc/ascend_kernels/prefill/sparse/kernel_body.h")
    receipt["compatibility_edits"].append({
        "file": "csrc/ascend_kernels/prefill/sparse/kernel_body.h",
        "reason": "Device bodies hidden from CANN9.1 host pass; identical device bodies, launch, schedule and parameters. Missing APIs copied verbatim from official CANN asc-devkit v9.2.0."})
    receipt["compatibility_sha256"] = hashes([Path(__file__), Path(__file__).parent / "cann91_api.h"])
    receipt["staged_sha256"] = hashes([staged / p.relative_to(VENDOR) for p in source_files()])
    includes = [staged / p.relative_to(VENDOR) if p.is_relative_to(VENDOR) else p for p in includes]
    constants = {}
    for node in ast.parse((VENDOR / "setup.py").read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("API_SOURCES", "ASCEND_KERNEL_SOURCES"):
            constants[node.targets[0].id] = ast.literal_eval(node.value)
    sources = constants["API_SOURCES"] + constants["ASCEND_KERNEL_SOURCES"]
    flags = ["-Ofast", "-std=c++20", "-fPIC", "-DNDEBUG", "-DTORCH_API_INCLUDE_EXTENSION_H",
             "-DTORCH_EXTENSION_NAME=npu", "-D_GLIBCXX_USE_CXX11_ABI=1",
             "--npu-arch=dav-3510", "-DFLASH_MLA_IS_BUILD_ON_ASCEND",
             "-mllvm", "-enable-hiipu-vf-loop-unroll", "--cce-res-usage",
             "-Wno-unneeded-internal-declaration", "-Wno-microsoft-template", "-Wno-deprecated-declarations",
             "-mllvm", "-cce-aicore-dcpreload-args=false"]
    for directory in includes:
        flags.extend(["-isystem", str(directory)])
    objects = []
    for number, source in enumerate(sources):
        obj = build_dir / (source.replace("/", "_") + ".o")
        log = build_dir / f"compile_{number:02d}.log"
        command = [compiler, *flags, "-c", str(staged / source), "-o", str(obj)]
        receipt["commands"].append({"argv": command, "log": str(log.relative_to(WORKSPACE))})
        save()
        with log.open("w") as stream:
            proc = subprocess.run(command, cwd=CACHE, stdout=stream, stderr=subprocess.STDOUT, timeout=600)
        receipt["commands"][-1]["returncode"] = proc.returncode
        if proc.returncode:
            return fail(f"Compiler failed for {source}; exact diagnostics: {log}")
        objects.append(obj)
    binary = build_dir / ("npu" + sysconfig.get_config_var("EXT_SUFFIX"))
    command = [compiler, "-shared", *map(str, objects), str(runtime), "-o", str(binary)]
    for libdir in [torch_root / "lib", npu_root / "lib", toolkit / "aarch64-linux/lib64"]:
        command += ["-L" + str(libdir), "-Wl,-rpath," + str(libdir)]
    command += ["-lc10", "-ltorch", "-ltorch_cpu", "-ltorch_python", "-ltorch_npu", "-lascendcl"]
    log = build_dir / "link.log"
    receipt["commands"].append({"argv": command, "log": str(log.relative_to(WORKSPACE))})
    save()
    with log.open("w") as stream:
        proc = subprocess.run(command, cwd=CACHE, stdout=stream, stderr=subprocess.STDOUT, timeout=600)
    receipt["commands"][-1]["returncode"] = proc.returncode
    if proc.returncode:
        return fail(f"Link failed; exact diagnostics: {log}")
    if hashes(source_files()) != receipt["source_sha256"]:
        return fail("Vendor source changed during compilation")
    receipt.update(status="compiled", binary=str(binary.relative_to(WORKSPACE)),
                   binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                   limitation="Compiled only; extension was never imported or executed")
    save()
    print(str(RECEIPT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
