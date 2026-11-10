"""Call the installed original topology discovery API under the host guard."""
import ctypes
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "sgl-topology-dependency01"
meta = json.loads(Path("/run/lock/ascend-npu/device-0.lock").read_text())
assert meta["phase"] == "running" and meta["child"]["pid"] == os.getpgrp()
assert meta["owner"] == "baseline-single-agent-local-20261027/sgl-topology-dependency01"
(OUT / "GUARD_RUNNING.json").write_text(json.dumps(meta, indent=2))
os.chdir(OUT)
row = {"diagnostic_only": True, "baseline_case_accepted": False}
for name in ("libdcmi.so",):
    try:
        ctypes.CDLL(name)
        row[name] = {"loadable": True}
    except OSError as exc:
        row[name] = {"loadable": False, "error": str(exc)}
path = Path("/usr/local/Ascend/cann-9.2.0-beta.2/lib64/libtopoaddrinfo.so").resolve()
row["topology_library"] = str(path)
row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
lib = ctypes.CDLL(str(path))
# Exact exported declaration in original topo_addr_info.c:
# int TopoAddrInfoGetTopoFilePath(int phyId, char *filePath, size_t bufSize)
fn = lib.TopoAddrInfoGetTopoFilePath
fn.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char), ctypes.c_size_t]
fn.restype = ctypes.c_int
buf = ctypes.create_string_buffer(4096)
row["topology_returncode"] = fn(0, buf, len(buf))
row["topology_path"] = buf.value.decode(errors="replace")
(OUT / "RESULT.json").write_text(json.dumps(row, indent=2) + "\n")
(OUT / "maps.txt").write_text(Path("/proc/self/maps").read_text())
print(json.dumps(row), flush=True)
