"""Existing shared ATVC Add factory; only bridge/cache paths relocated to owned scope."""
from pathlib import Path
from benchmarks.ascend_baselines import Candidate, LIBRARIES, WORKSPACE, UnsupportedCase, _add_layout, _owned_native, source_hashes

def atvc_add(case):
    import ctypes as C
    import hashlib
    import os
    import shutil
    import subprocess
    import torch
    from benchmarks.api import Implementation

    x, _, shape, _, _, pack = _add_layout(case)
    suffixes = {torch.float16: "f16", torch.bfloat16: "bf16", torch.float32: "f32"}
    if x.dtype not in suffixes:
        raise UnsupportedCase("The existing AddCustom template is bound for FP16/BF16/FP32.")
    symbol = "classic_atvc_add_" + suffixes[x.dtype]
    root = LIBRARIES / "atvc"
    bridge = WORKSPACE / "TileOPs/benchmarks/native/ascend/atvc_add.cpp"
    headers = sorted((root / "include").rglob("*.h"))
    host_header = root / "examples/ops_pytorch/add/add_custom_impl.h"
    hashes = source_hashes(Path(__file__), bridge, host_header, *headers)
    compiler = shutil.which("bisheng")
    if compiler is None:
        raise RuntimeError("The installed Ascend compiler is unavailable.")
    digest = hashlib.sha256(repr(sorted(hashes.items())).encode()).hexdigest()
    cache = WORKSPACE / ".cache/all-pairs/atvc" / ("atvc-add-" + digest[:20])
    cache.mkdir(parents=True, exist_ok=True)
    binary = cache / "libclassic_atvc_add.so"
    if not binary.exists():
        sdk = Path(os.environ["ASCEND_HOME_PATH"])
        temporary = cache / "libclassic_atvc_add.building.so"
        command = [compiler, "-x", "asc", "--npu-arch=dav-3510",
                   "-shared", "-fPIC", str(bridge), "-o", str(temporary),
                   "-I", str(root / "include"), "-I", str(host_header.parent),
                   "-L", str(sdk / "lib64"), "-ltiling_api", "-lplatform",
                   "-lascendcl", "-lm", "-ldl"]
        result = subprocess.run(command, capture_output=True, text=True)
        (cache / "build.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f"ATVC original-template compilation failed: {cache / 'build.log'}")
        temporary.replace(binary)
    lib = _owned_native(binary, [symbol])
    native_add = getattr(lib, symbol)
    native_add.argtypes = [
        C.c_void_p, C.c_void_p, C.c_void_p, C.c_int64, C.c_void_p,
    ]
    native_add.restype = C.c_int
    out = torch.empty(shape, dtype=x.dtype, device=x.device)

    def run(x, y):
        a, b = pack(x), pack(y)
        rc = native_add(
            a.data_ptr(), b.data_ptr(), out.data_ptr(), out.numel(),
            torch.npu.current_stream().npu_stream)
        if rc:
            raise RuntimeError(f"ATVC host bridge returned {rc}")
        return out

    return Candidate(
        Implementation(run), "ATVC::AddCustom<" + str(x.dtype) + ">",
        "Instantiate original generic Add template; thin C ABI; original tiling and buffer defaults.",
        hashes | source_hashes(binary),
    )
