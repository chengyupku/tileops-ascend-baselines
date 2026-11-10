#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
cumsum batch 一致性验证脚本。

判据：同一样本（固定种子生成）嵌入不同 batch 大小 / 不同位置，逐 bit 比较输出切片。
  - level=0（默认模式）：预期【不一致】——当前版本算子无 batch 一致性实现，用于证明测试有区分度；
  - level=3（BI 模式）：预期【一致】——开发完成后的验收门禁。

用法：
  python3 verify_batch_invariance.py [--level 0|3] [--dtypes float32 float16]
      [--lens 4096 262144] [--batches 1 2 3 5 8 63 64 65] [--attrs base]
      [--device 0] [--fail-fast]
"""

import argparse
import ctypes
import os
import sys

import numpy as np
import torch
import torch_npu  # noqa: F401  # noqa: F401  注册 npu backend

# ---------------------------------------------------------------------------
# aclrtSetSysParamOpt：确定性开关（进程级，必须在 device 交互前设置）
# ---------------------------------------------------------------------------
ACL_OPT_DETERMINISTIC = 0  # aclSysParamOpt 枚举（acl_base_rt.h:208），value 才是级别
# value: 0=默认  1/2=确定性计算  3=batch 一致性（reduce_sum 口径）

_libruntime = None


def _load_runtime():
    """从 CANN 安装路径加载 libruntime，返回 ctypes 句柄。"""
    global _libruntime
    if _libruntime is not None:
        return _libruntime
    ascend_home = os.environ.get(
        "ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest"
    )
    for cand in (
        os.path.join(ascend_home, "lib64", "libascendcl.so"),
        os.path.join(ascend_home, "lib64", "libruntime.so"),
        os.path.join(ascend_home, "runtime", "lib64", "libruntime.so"),
    ):
        if os.path.exists(cand):
            _libruntime = ctypes.CDLL(cand)
            return _libruntime
    raise RuntimeError(f"libruntime.so not found under {ascend_home}")


def set_deterministic_level(level: int) -> bool:
    """调用 aclrtSetSysParamOpt(ACL_OPT_DETERMINISTIC, level)。

    返回 True 表示调用成功；若环境不支持（旧 CANN 无此接口/枚举不符）返回 False 并打印告警。
    注意：该接口需在 NPU 任何初始化/分配之前调用（脚本中在 import torch_npu 后、
    首次 to(npu) 前由 main 流程保证顺序，见 main() 中说明）。
    """
    rt = _load_runtime()
    try:
        fn = rt.aclrtSetSysParamOpt
    except AttributeError:
        print(
            f"[WARN] aclrtSetSysParamOpt not found in libruntime (level={level} 不生效，走默认路径)"
        )
        return False
    fn.argtypes = [ctypes.c_int, ctypes.c_int64]
    fn.restype = ctypes.c_int32
    ret = fn(ACL_OPT_DETERMINISTIC, level)
    if ret != 0:
        print(
            f"[WARN] aclrtSetSysParamOpt(ACL_OPT_DETERMINISTIC={ACL_OPT_DETERMINISTIC}, {level}) ret={ret}"
        )
        return False
    return True


# ---------------------------------------------------------------------------
# 用例构造与执行
# ---------------------------------------------------------------------------
DTYPES = {
    "float32": (torch.float32, np.float32),
    "float16": (torch.float16, np.float16),
    "bfloat16": (torch.bfloat16, None),  # numpy 无 bf16，比较走 torch
}

SLOT_POS = {"head": lambda b: 0, "mid": lambda b: b // 2, "tail": lambda b: b - 1}


def make_sample(len_r: int, dtype: str, seed: int):
    """固定种子构造“易漂移”样本：小量级+大量级混合，放大浮点结合序差异。"""
    rng = np.random.default_rng(seed)
    if dtype == "bfloat16":
        g = torch.Generator().manual_seed(seed)
        t = torch.randn(len_r, dtype=torch.float32, generator=g)
        scale = torch.where(torch.arange(len_r) % 3 == 0, 1e3, 1e-3)
        return (t * scale).to(torch.bfloat16)
    base = rng.standard_normal(len_r).astype(DTYPES[dtype][1])
    scale = np.where(np.arange(len_r) % 3 == 0, 1e3, 1e-3).astype(DTYPES[dtype][1])
    return (base * scale).astype(DTYPES[dtype][1])


def make_batch(
    sample: np.ndarray, batch: int, slot: int, dtype: str, seed: int
) -> torch.Tensor:
    """把样本嵌入 batch 指定位置，其余行用不同种子的填充数据。"""
    if dtype == "bfloat16":
        g = torch.Generator().manual_seed(seed + 1)
        filled = torch.randn(batch, sample.shape[0], dtype=torch.float32, generator=g)
        scale = torch.where(torch.arange(batch) % 2 == 0, 1.0, 100.0).unsqueeze(1)
        filled = (filled * scale).to(torch.bfloat16)
        filled[slot] = sample
        return filled
    filled = (
        np.random.default_rng(seed + 1)
        .standard_normal((batch, sample.shape[0]))
        .astype(DTYPES[dtype][1])
    )
    scale = np.where(np.arange(batch)[:, None] % 2 == 0, 1.0, 100.0).astype(
        DTYPES[dtype][1]
    )
    filled = (filled * scale).astype(DTYPES[dtype][1])
    filled[slot] = sample
    return torch.from_numpy(filled)


def run_cumsum(
    x: torch.Tensor, dim: int, exclusive: bool, reverse: bool, device: str
) -> np.ndarray:
    """aclnnCumsumV2 直调（vendor BI 路径）。

    注意：torch.cumsum 经 torch_npu 路由到 aclnnCumsum(V1 内置实现)，不进 vendor BI
    路径且不吃 deterministic 注入，其跨 M 差异与本判据无关——必须走 V2 直调。
    """
    global _vendor, _aclrtMalloc, _aclrtFree
    x_np = x.numpy() if isinstance(x, torch.Tensor) and x.dtype != torch.bfloat16 else x
    # bf16 走 torch 路径创建（numpy 不支持），读回时以 uint16 位视图返回（bitwise 判据不受影响）
    is_bf16 = isinstance(x, torch.Tensor) and x.dtype == torch.bfloat16
    t_in = (
        _iface.create_acl_tensor(x, "ND")
        if is_bf16
        else _iface.create_acl_tensor(x_np, "ND")
    )
    out_holder = (
        torch.empty(tuple(x.shape), dtype=torch.bfloat16)
        if is_bf16
        else np.zeros_like(x_np)
    )
    t_out = _iface.create_acl_tensor(out_holder, "ND")
    exe = ctypes.c_void_p()
    ws = ctypes.c_uint64(0)
    assert (
        _vendor.aclnnCumsumV2GetWorkspaceSize(
            t_in, dim, exclusive, reverse, t_out, ctypes.byref(ws), ctypes.byref(exe)
        )
        == 0
    )
    wsPtr = ctypes.c_void_p()
    if ws.value > 0:
        assert _aclrtMalloc(ctypes.byref(wsPtr), ws.value) == 0
    assert _vendor.aclnnCumsumV2(wsPtr, ws.value, exe, _stream) == 0
    _iface._rts_interface.synchronize_with_stream(_stream)
    if wsPtr.value:
        _aclrtFree(wsPtr)
    nbytes = x.element_size() * x.numel() if is_bf16 else x_np.nbytes
    out = (
        np.frombuffer(
            _iface.get_data_from_hbm(_iface.get_device_mem_addr(t_out), nbytes),
            dtype=np.uint16 if is_bf16 else x_np.dtype,
        )
        .reshape(x.shape)
        .copy()
    )
    _iface._free_acl_tensor(t_in)
    _iface._free_acl_tensor(t_out)
    return out


def bitwise_equal(a, b) -> bool:
    a_np = a.cpu().numpy() if isinstance(a, torch.Tensor) else a
    b_np = b.cpu().numpy() if isinstance(b, torch.Tensor) else b
    if a_np.dtype != b_np.dtype:
        return False
    if a_np.dtype == np.bool_:
        return np.array_equal(a_np, b_np)
    return np.array_equal(a_np.view(np.uint8), b_np.view(np.uint8))


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--level",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="确定性级别（0=默认非BI，3=BI，验证脚本在当前版本应跑 0 观察不一致）",
    )
    ap.add_argument("--dtypes", nargs="+", default=["float32"])
    ap.add_argument(
        "--lens",
        nargs="+",
        type=int,
        default=[4096],
        help="序列长度网格；覆盖 B/UB 边界与长序列",
    )
    ap.add_argument(
        "--batches", nargs="+", type=int, default=[1, 2, 3, 5, 8, 63, 64, 65]
    )
    ap.add_argument(
        "--attrs",
        nargs="+",
        default=["base"],
        choices=["base", "exclusive", "reverse", "exrev"],
    )
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fail-fast", action="store_true")
    ap.add_argument(
        "--allow-builtin",
        action="store_true",
        help="允许 vendor 缺失时 fallback 到 builtin 实现（默认拒绝，防虚假验证）",
    )
    args = ap.parse_args()

    device = f"npu:{args.device}"
    print("=== cumsum batch-invariance verify ===")
    print(
        f"level={args.level}  device={device}  dtypes={args.dtypes}  lens={args.lens}"
    )
    print(f"batches={args.batches}  attrs={args.attrs}  seed={args.seed}")

    ok = set_deterministic_level(args.level)
    print(
        f"aclrtSetSysParamOpt(ACL_OPT_DETERMINISTIC, {args.level}) -> {'OK' if ok else 'FAILED(默认路径)'}"
    )

    global _iface, _stream, _vendor, _aclrtMalloc, _aclrtFree
    asc = os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_TOOLKIT_HOME")
    from ttk.core_modules.aclnn.acl_interface import AclInterface

    _iface = AclInterface("ascend950pr", False)
    _iface.set_device(args.device)
    _stream = _iface.create_stream()

    def _find_opapi(require_vendor):
        import glob as _glob

        opp = os.environ.get("ASCEND_OPP_PATH")
        asc = os.environ.get("ASCEND_HOME_PATH") or os.environ.get(
            "ASCEND_TOOLKIT_HOME"
        )
        vendor_hits = []
        if opp:
            for p in sorted(_glob.glob(f"{opp}/vendors/*/op_api/lib/libcust_opapi.so")):
                try:
                    lib = ctypes.CDLL(p)
                except OSError:
                    continue
                if hasattr(lib, "aclnnCumsumV2"):
                    vendor_hits.append(p)
        if vendor_hits:
            print(f"opapi: {vendor_hits[0]} (vendor)")
            return ctypes.CDLL(vendor_hits[0])
        if require_vendor:
            # vendor 缺失时 builtin 会静默接管，测到的不是被测实现 → 拒绝执行（防虚假验证）
            raise RuntimeError(
                "vendor libcust_opapi.so 缺失或不含 aclnnCumsumV2，拒绝 fallback 到 builtin。"
                "请先部署 vendor 包；确要验证 builtin 实现时显式传 --allow-builtin。"
            )
        for p in (
            [f"{asc}/lib64/libopapi.so", f"{asc}/opp/built-in/op_api/lib/libopapi.so"]
            if asc
            else []
        ):
            if not os.path.exists(p):
                continue
            try:
                lib = ctypes.CDLL(p)
            except OSError:
                continue
            if hasattr(lib, "aclnnCumsumV2"):
                print(f"opapi: {p} (builtin, --allow-builtin)")
                return lib
        raise RuntimeError("aclnnCumsumV2 not found in any candidate opapi library")

    _vendor = _find_opapi(not args.allow_builtin)
    _vendor.aclnnCumsumV2GetWorkspaceSize.restype = ctypes.c_int
    _vendor.aclnnCumsumV2GetWorkspaceSize.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    _vendor.aclnnCumsumV2.restype = ctypes.c_int
    _vendor.aclnnCumsumV2.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    _acl = ctypes.CDLL(f"{asc}/lib64/libascendcl.so")
    _aclrtMalloc = _acl.aclrtMalloc
    _aclrtMalloc.restype = ctypes.c_int
    _aclrtMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint64]
    _aclrtFree = _acl.aclrtFree

    total = 0
    consistent = 0
    inconsistent = 0
    rows = []
    for dtype in args.dtypes:
        for len_r in args.lens:
            for attr in args.attrs:
                exclusive = attr in ("exclusive", "exrev")
                reverse = attr in ("reverse", "exrev")
                sample = make_sample(len_r, dtype, args.seed)
                ref = None
                ref_cfg = None
                for batch in args.batches:
                    for slot_name, slot_fn in SLOT_POS.items():
                        slot = slot_fn(batch)
                        x = make_batch(
                            sample, batch, slot, dtype, args.seed + batch * 10 + slot
                        )
                        out = run_cumsum(
                            x,
                            dim=1,
                            exclusive=exclusive,
                            reverse=reverse,
                            device=device,
                        )
                        got = out[slot] if hasattr(out, "__getitem__") else out
                        if ref is None:
                            # 独立 golden 对照：首算行与 numpy 参考比对，拦截"执行失败输出全零"
                            # 或"fallback 到非被测实现"等形态（这些形态下跨 M bitwise 仍可全 SAME）
                            x_ref = x  # bf16 无 numpy 表示，统一走 torch 域 golden
                            if reverse:  # 反向累加 = flip(cumsum(flip(x)))
                                x_ref = torch.flip(x_ref, dims=[1])
                            g = torch.cumsum(x_ref.to(torch.float32), dim=1)
                            if exclusive:  # exclusive 在翻转域内 shift
                                z = torch.zeros_like(g[:, :1])
                                g = torch.cat([z, g[:, :-1]], dim=1)
                            if reverse:
                                g = torch.flip(g, dims=[1])
                            out_tdt = {
                                "float32": torch.float32,
                                "float16": torch.float16,
                                "bfloat16": torch.bfloat16,
                            }.get(dtype, torch.float32)
                            # golden 经输出域量化（fp16 溢出 inf 与 kernel 同款行为），同号 inf 差记 0
                            g_row = g[slot].to(out_tdt).to(torch.float32).numpy()
                            got_np = (
                                got.cpu().numpy()
                                if isinstance(got, torch.Tensor)
                                else got
                            )
                            if got_np.dtype == np.uint16:  # bf16 位视图 → 数值
                                got_np = (
                                    torch.from_numpy(got_np)
                                    .view(torch.bfloat16)
                                    .to(torch.float32)
                                    .numpy()
                                )
                            # 异常检测阈值取 golden 量级的 10%：只拦"输出未被写入/通道异常"级
                            # 偏差（全零时差≈golden 全量级），对消样本的正常树序差不误伤
                            g_mag = (
                                float(np.nan_to_num(np.max(np.abs(g_row)), posinf=1e30))
                                + 1.0
                            )
                            diff = got_np.astype(np.float32) - g_row
                            same_inf = np.isinf(diff)
                            diff = np.where(
                                same_inf,
                                0.0,
                                np.nan_to_num(diff, posinf=0.0, neginf=0.0),
                            )
                            d_max = float(np.max(np.abs(diff))) if diff.size else 0.0
                            if d_max > 0.1 * g_mag:
                                print(
                                    f"[GOLDEN-MISMATCH] {dtype} L={len_r} {attr} B={batch}/{slot_name} "
                                    f"max_abs={d_max:.3e} golden_mag={g_mag:.3e} —— "
                                    f"输出疑似未被写入（全零）或执行通道异常，验证中止"
                                )
                                sys.exit(2)
                            ref, ref_cfg = got, f"B={batch}/{slot_name}"
                            continue
                        total += 1
                        if bitwise_equal(got, ref):
                            consistent += 1
                            rows.append(
                                (dtype, len_r, attr, batch, slot_name, "SAME", "")
                            )
                        else:
                            inconsistent += 1
                            g_np = (
                                got.cpu().numpy()
                                if isinstance(got, torch.Tensor)
                                else got
                            )
                            r_np = (
                                ref.cpu().numpy()
                                if isinstance(ref, torch.Tensor)
                                else ref
                            )
                            n_diff = int(
                                np.sum(g_np.view(np.uint8) != r_np.view(np.uint8))
                            )
                            rows.append(
                                (
                                    dtype,
                                    len_r,
                                    attr,
                                    batch,
                                    slot_name,
                                    "DIFF",
                                    f"{n_diff} bytes differ (vs {ref_cfg})",
                                )
                            )
                            if args.fail_fast:
                                print_report(
                                    rows, total, consistent, inconsistent, args.level
                                )
                                sys.exit(2 if args.level == 0 else 1)

    print_report(rows, total, consistent, inconsistent, args.level)
    # 退出码语义：
    #   level=0（当前版本）：预期不一致 -> 有 DIFF 才证明测试有区分度 -> DIFF 存在返回 0，全 SAME 返回 3（可疑）
    #   level=3（BI 版）：   预期一致   -> 全 SAME 返回 0，出现 DIFF 返回 1（门禁失败）
    has_diff = inconsistent > 0
    if args.level == 0:
        sys.exit(0 if has_diff else 3)
    sys.exit(0 if not has_diff else 1)


def print_report(rows, total, consistent, inconsistent, level):
    print("\n--- report ---")
    print(
        f"{'dtype':<10}{'lenR':>8}{'attr':<10}{'batch':>6}{'slot':<6}{'result':<7}detail"
    )
    for r in rows:
        print(f"{r[0]:<10}{r[1]:>8}{r[2]:<10}{r[3]:>6}{r[4]:<6}{r[5]:<7}{r[6]}")
    print(
        f"\ncomparisons={total}  bitwise-same={consistent}  bitwise-diff={inconsistent}"
    )
    if level == 0:
        print(
            "VERDICT(level=0, 当前版本): "
            + (
                "PASS —— 观察到不一致，证明测试有区分度，等待 BI 实现"
                if inconsistent
                else "SUSPECT —— 全部一致，需检查用例是否真的触发了不同 tiling 路径"
            )
        )
    else:
        print(
            "VERDICT(level=3, BI 门禁): "
            + (
                "PASS —— batch 一致"
                if not inconsistent
                else "FAIL —— 存在 batch 相关的 bit 漂移"
            )
        )


if __name__ == "__main__":
    main()
