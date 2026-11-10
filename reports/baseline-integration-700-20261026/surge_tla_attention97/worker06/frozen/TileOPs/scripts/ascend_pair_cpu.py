#!/usr/bin/env python3
"""CPU bootstrap: python -I -S -B ascend_pair_cpu.py --site-packages PATH -- ...

The supplied directory must hold a CPU-only Torch installation. It is appended
as a plain path; site.addsitedir and .pth processing are deliberately not used.
"""
import argparse
import importlib.abc
import importlib.util
import os
from pathlib import Path
import runpy
import sys


def main():
    if not sys.flags.no_site or not sys.flags.isolated or not sys.dont_write_bytecode:
        raise RuntimeError("CPU phases require python -I -S -B (no site/.pth startup)")
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-packages", type=Path, required=True)
    parser.add_argument("--dependency-site-packages", type=Path, action="append", default=[])
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    os.environ.update(TORCH_DEVICE_BACKEND_AUTOLOAD="0", FLA_NPU_DISABLE_PTH="1",
                      PYTHONDONTWRITEBYTECODE="1", ASCEND_PAIR_CPU_ISOLATED="1")
    blocked = {"torch_npu", "acl", "aclnn", "pyacl", "torchair", "triton", "tilelang",
               "tvm", "vllm", "vllm_ascend", "fla", "cann", "tbe", "te", "ascend"}

    class DenyDevice(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in blocked:
                raise ImportError("CPU phase forbids device runtime import: " + fullname)

    sys.meta_path.insert(0, DenyDevice())
    find_spec = importlib.util.find_spec
    def optional_spec(name, *a, **kw):
        return None if name.split(".")[0] in blocked else find_spec(name, *a, **kw)
    importlib.util.find_spec = optional_spec
    here = Path(__file__).resolve().parent
    sys.path[:] = [p for p in sys.path if "site-packages" not in p and "Ascend" not in p]
    sys.path.extend([str(args.site_packages.resolve()),
                     *(str(p.resolve()) for p in args.dependency_site_packages), str(here)])
    def audit(event, values):
        if event == "ctypes.dlopen" and values and values[0]:
            name = str(values[0]).lower()
            if any(token in name for token in ("ascend", "libacl", "torch_npu", "davinci", "libcann", "libdrv", "libdsmi", "libaicpu")):
                raise RuntimeError("CPU phase forbids runtime library: " + name)
        if event == "open" and values and isinstance(values[0], (str, bytes)):
            name = os.fsdecode(values[0])
            if name.startswith(("/dev/davinci", "/dev/devmm", "/dev/hisi", "/dev/nvidia")):
                raise RuntimeError("CPU phase forbids device file: " + name)
    sys.addaudithook(audit)
    import torch
    if not torch.__version__.endswith("+cpu"):
        raise RuntimeError("select an isolated CPU-only Torch installation")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    torch.set_default_device("cpu")
    rest = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    sys.argv = [str(here / "ascend_pair_phases.py"), *rest]
    try:
        runpy.run_path(sys.argv[0], run_name="__main__")
    finally:
        bad = [n for n in sys.modules if n.split(".")[0] in blocked]
        maps = Path("/proc/self/maps").read_text()
        libraries = [line for line in maps.splitlines()
                     if any(token in line.lower() for token in ("libascend", "libacl", "libtorch_npu", "libcann", "libdrv", "libdsmi", "libaicpu"))]
        if bad or libraries:
            raise RuntimeError("CPU phase loaded forbidden runtime: " + repr((bad, libraries)))
        print("CPU_PHASE_SAFETY", {"torch": torch.__version__, "no_site": True,
              "backend_autoload": os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"],
              "forbidden_modules": bad, "runtime_libraries": libraries}, file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
