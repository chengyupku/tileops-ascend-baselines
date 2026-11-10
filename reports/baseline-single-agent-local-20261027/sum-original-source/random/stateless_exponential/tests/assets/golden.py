#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

__spec__ = {
    "aclnnStatelessExponentialTensor": "aclnnStatelessExponentialTensorSpec",
}

import torch
import numpy as np

_PHILOX_W32_A = 0x9E3779B9
_PHILOX_W32_B = 0xBB67AE85
_PHILOX_M4X32_A = 0xD2511F53
_PHILOX_M4X32_B = 0xCD9E8D57
_U32 = 0xFFFFFFFF
_U64 = 0xFFFFFFFFFFFFFFFF
_RAND_2POW32_INV = np.float32(2.3283064e-10)
_RAND_2POW32_INV_HALF = np.float32(2.3283064e-10 / 2.0)
_SAMPLES_ALIGNMENT = 128
_UNROLL = 4
_SIMT_THREAD_GROUP_SIZE = 256
_MAX_THREADS_PER_AIC = 2048
_AIC_CLUSTER_COUNT = 78

_torch_npu = None

_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "half": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
    "float": torch.float32,
}


def _as_torch(value):
    if isinstance(value, torch.Tensor):
        return value
    if value.dtype.name == "bfloat16":
        return torch.frombuffer(
            bytearray(value.tobytes()), dtype=torch.bfloat16
        ).reshape(value.shape)
    return torch.from_numpy(np.ascontiguousarray(value))


def _philox_l0rounds(c0, c1, c2, c3, seed):
    seedu = int(seed) & _U64
    k0, k1 = seedu & _U32, (seedu >> 32) & _U32
    ks = []
    for _ in range(10):
        ks.append((np.uint32(k0), np.uint32(k1)))
        k0 = (k0 + _PHILOX_W32_A) & _U32
        k1 = (k1 + _PHILOX_W32_B) & _U32
    a = np.uint64(_PHILOX_M4X32_A)
    b = np.uint64(_PHILOX_M4X32_B)
    for rk0, rk1 in ks:
        p0 = c0.astype(np.uint64) * a
        lo0 = p0.astype(np.uint32)
        hi0 = (p0 >> np.uint64(32)).astype(np.uint32)
        p1 = c2.astype(np.uint64) * b
        lo1 = p1.astype(np.uint32)
        hi1 = (p1 >> np.uint64(32)).astype(np.uint32)
        c0, c1, c2, c3 = (hi1 ^ c1 ^ rk0, lo1, hi0 ^ c3 ^ rk1, lo0)
    return c0, c1, c2, c3


def _get_torch_npu():
    global _torch_npu
    if _torch_npu is None:
        import torch_npu

        _torch_npu = torch_npu
    return _torch_npu


def _offset_div4_u64(off):
    return (((int(off) & _U64) + (_UNROLL - 1)) & _U64) // _UNROLL


def _to_compute_tensor(probs, dtype_str):
    return probs.detach().to(torch.float32).to(_DTYPE_MAP.get(dtype_str, torch.float32))


def _no_replacement_path(probs, dtype_str, lambdaV, seed, real_offset):
    w = _to_compute_tensor(probs, dtype_str)
    was_1d = w.ndim == 1
    if was_1d:
        w = w.unsqueeze(0)
    output_size = w.numel()

    grid = max(
        (output_size + _SIMT_THREAD_GROUP_SIZE - 1) // _SIMT_THREAD_GROUP_SIZE, 1
    )
    grid = min(
        _AIC_CLUSTER_COUNT * (_MAX_THREADS_PER_AIC // _SIMT_THREAD_GROUP_SIZE), grid
    )
    total_threads = grid * _SIMT_THREAD_GROUP_SIZE

    base_offset = _offset_div4_u64(real_offset)
    li = np.arange(output_size, dtype=np.int64)
    liner = li % total_threads
    q = li // total_threads
    istep = (q % _UNROLL).astype(np.int64)
    loop = q // _UNROLL
    lower = np.uint64(base_offset & _U64) + loop.astype(np.uint64)
    upper = liner.astype(np.uint64)
    c0 = (lower & _U32).astype(np.uint32)
    c1 = ((lower >> np.uint64(32)) & _U32).astype(np.uint32)
    c2 = (upper & _U32).astype(np.uint32)
    c3 = ((upper >> np.uint64(32)) & _U32).astype(np.uint32)
    r0, r1, r2, r3 = _philox_l0rounds(c0, c1, c2, c3, seed)
    res = np.stack([r0, r1, r2, r3], axis=1)
    chosen = res[np.arange(output_size), istep]
    u = chosen.astype(np.float32) * _RAND_2POW32_INV + _RAND_2POW32_INV_HALF

    half_eps = float(np.float32(1.1920929e-07 / 2.0))
    u_tensor = torch.from_numpy(u)
    log_v = torch.where(
        u_tensor >= 1.0 - half_eps,
        torch.tensor(-half_eps, dtype=torch.float32),
        torch.log(u_tensor),
    )
    lamVal = -1 / lambdaV
    out = (lamVal * log_v).to(dtype_str)
    return out


class ThirdPartyImpl:
    def __init__(self, *args, **kwargs):
        self_t, seed_t, offset_t, offset, lambd = (list(args) + [None] * 5)[:5]
        self.offset = offset_t[0].item() + offset
        self.lambd = lambd
        self.count = self_t.numel()
        self.seed = seed_t[0].item()
        self.dtype = self_t.dtype

    def __call__(self, **kwargs):
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        torch.cuda.manual_seed(self.seed)
        default_gen = torch.cuda.default_generators[0]
        if hasattr(default_gen, "set_offset"):
            default_gen.set_offset(self.offset)
        return torch.empty(self.count, device=device, dtype=self.dtype).exponential_(
            self.lambd
        )


def compute_data(
    self,
    seedTensor,
    offsetTensor,
    offset,
    lambd,
):
    self_tensor = _as_torch(self).detach().cpu()
    seed = int(_as_torch(seedTensor).reshape(-1)[0].item())
    offset2 = int(_as_torch(offsetTensor).reshape(-1)[0].item())
    outDtype = self.dtype
    realOffset = offset + offset2
    result = _no_replacement_path(self_tensor, outDtype, lambd, seed, realOffset)

    return result


class aclnnStatelessExponentialTensorSpec:
    """ACLNN 流程 — golden / third_party 均收到 torch.Tensor（已在设备上）"""

    def golden(
        self,
        seedTensor,
        offsetTensor,
        offset,
        lambd,
        **kwargs,
    ):
        return compute_data(self, seedTensor, offsetTensor, offset, lambd)

    third_party = {"torch": ThirdPartyImpl}
    tolerance = {
        "float32": {"standard": "cross_check", "level": "L1"},
        "float16": {"standard": "cross_check", "level": "L1"},
        "bfloat16": {"standard": "cross_check", "level": "L1"},
    }
