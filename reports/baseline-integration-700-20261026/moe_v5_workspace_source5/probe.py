#!/usr/bin/env python3
"""Parent-only V5 workspace query probe; never run in this worker.

The module has no torch/NPU import at import time. Parent must run one case per
fresh process under npu-guard and may set the absolute vendor library path.
"""
from __future__ import annotations
import argparse, ctypes as C, importlib, json, os, sys, time
from pathlib import Path

CASES = {
    "decode_down": (16, 7168, 2048),
    "decode_down_fp32": (16, 7168, 2048),
    "decode_up": (16, 4096, 7168),
    "prefill_down": (128, 7168, 2048),
    "prefill_up": (128, 4096, 7168),
}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--case", required=True, choices=CASES); ap.add_argument("--out", default="probe_result.json")
    a = ap.parse_args()
    if os.environ.get("ALLOW_PARENT_NPU_PROBE") != "1":
        raise SystemExit("refusing to run: parent must set ALLOW_PARENT_NPU_PROBE=1 after NPU release")
    torch = importlib.import_module("torch")
    if not getattr(torch, "npu", None) or not torch.npu.is_available(): raise RuntimeError("NPU unavailable")
    M, N, K = CASES[a.case]; dev = torch.device("npu")
    x, w, y = (torch.empty(s, device=dev, dtype=torch.float32) for s in ((M,K),(N,K),(M,N)))
    lib_path = os.environ["ASCEND_GMM_LIB"]
    lib = C.CDLL(lib_path); base = C.CDLL(os.environ.get("ASCEND_NNOPBASE_LIB", "libnnopbase.so"))
    # Parent should copy the proven aclCreateTensor/aclCreateTensorList bridge from ascend_math_more.py.
    raise RuntimeError("descriptor bridge intentionally requires parent integration; no unsafe fake pointers")

if __name__ == "__main__": main()
