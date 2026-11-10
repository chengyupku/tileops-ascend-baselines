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
"""
drop_out_v3
"""

import copy
import numpy as np
from functools import reduce
from typing import List
import torch

try:
    from ml_dtypes import bfloat16 as _bf16
except ImportError:
    _bf16 = np.float32

__spec__ = {
    "aclnnDropoutV3Tensor": "aclnnDropoutV3TensorSpec",
    "aclnnDropoutV3": "aclnnDropoutV3Spec",
}

MASK_32 = 0xFFFFFFFF
PHILOX_M4_32 = [0xD2511F53, 0xCD9E8D57]
PHILOX_W_32 = [0x9E3779B9, 0xBB67AE85]
VAL_1, VAL_2, VAL_3, VAL_4 = 0, 1, 2, 3

M0 = np.uint64(0xD2511F53)
M1 = np.uint64(0xCD9E8D57)
W0 = np.uint64(0x9E3779B9)
W1 = np.uint64(0xBB67AE85)
MASK64 = np.uint64(0xFFFFFFFF)
RAND_2POW32_INV = np.float32(2.3283064e-10)
RAND_2POW32_INV_HALF = np.float32(RAND_2POW32_INV / 2.0)

BLOCK_SIZE = 256
MAX_THREADS_PER_AIC = 2048
AIC_CLUSTER_COUNT = 78


def _as_torch(value):
    if isinstance(value, torch.Tensor):
        return value
    if value.dtype.name == "bfloat16":
        return torch.frombuffer(
            bytearray(value.tobytes()), dtype=torch.bfloat16
        ).reshape(value.shape)
    return torch.from_numpy(np.ascontiguousarray(value))


def mask_to_align16(uint8_values):
    padding = 16 - (len(uint8_values) % 16)
    if padding != 16:
        uint8_values = np.pad(
            uint8_values, (0, padding), mode="constant", constant_values=0
        )
    return uint8_values


def philox_10_rounds_vec(counters, key):
    """向量化 Philox 4x32-10 轮运算

    counters: (N, 4) uint64 数组
    key: [k0, k1] 标量
    返回: (N, 4) uint64 数组
    """
    c0 = counters[:, 0].copy()
    c1 = counters[:, 1].copy()
    c2 = counters[:, 2].copy()
    c3 = counters[:, 3].copy()
    k0 = np.uint64(key[0])
    k1 = np.uint64(key[1])

    for _ in range(9):
        p0 = M0 * c0
        hi0 = p0 >> np.uint64(32)
        lo0 = p0 & MASK64
        p1 = M1 * c2
        hi1 = p1 >> np.uint64(32)
        lo1 = p1 & MASK64
        c0 = (hi1 ^ c1 ^ k0) & MASK64
        c1 = lo1
        c2 = (hi0 ^ c3 ^ k1) & MASK64
        c3 = lo0
        k0 = (k0 + W0) & MASK64
        k1 = (k1 + W1) & MASK64

    p0 = M0 * c0
    hi0 = p0 >> np.uint64(32)
    lo0 = p0 & MASK64
    p1 = M1 * c2
    hi1 = p1 >> np.uint64(32)
    lo1 = p1 & MASK64
    c0 = (hi1 ^ c1 ^ k0) & MASK64
    c1 = lo1
    c2 = (hi0 ^ c3 ^ k1) & MASK64
    c3 = lo0

    result = np.stack([c0, c1, c2, c3], axis=1)
    return result


def gen_key_and_counter(threadIdx: int, seed: int, offset: int):
    key_ = [0] * 2
    counter_ = [0] * 4
    key_[0] = seed & MASK_32
    key_[1] = (seed >> 32) & MASK_32
    counter_[2] = threadIdx & MASK_32
    counter_[3] = (threadIdx >> 32) & MASK_32
    counter_[0] = offset & MASK_32
    counter_[1] = (offset >> 32) & MASK_32
    return key_, counter_


def philox(rounds: int, counter: List, key: List, count: int):
    num_calls = (count + 255) // 256 * 256 // 4
    counters = np.zeros((num_calls, 4), dtype=np.uint64)
    c = [
        counter[0] & MASK_32,
        counter[1] & MASK_32,
        counter[2] & MASK_32,
        counter[3] & MASK_32,
    ]
    for i in range(num_calls):
        counters[i] = [c[0], c[1], c[2], c[3]]
        c[0] = (c[0] + 1) & MASK_32
        if c[0] == 0:
            c[1] = (c[1] + 1) & MASK_32
            if c[1] == 0:
                c[2] = (c[2] + 1) & MASK_32
                if c[2] == 0:
                    c[3] = (c[3] + 1) & MASK_32

    result = philox_10_rounds_vec(counters, key)
    ret = result.astype(np.uint32).flatten()[:count]
    return ret


def curand_uniform(x):
    RAND_2POW32_INV = np.float32(2.3283064365386963e-10)
    RAND_2POW32_INV_HALF = np.float32(RAND_2POW32_INV / np.float32(2.0))
    return x.astype(np.float32) * RAND_2POW32_INV + RAND_2POW32_INV_HALF


def update_prob_type(prob, dtype):
    if dtype in ["bfloat16", "bfloat16_t"]:
        return np.array(prob).astype(_bf16).astype(np.cfloat).astype(_bf16).item()
    elif dtype in ["half", "float16", "float16_t"]:
        return np.array(prob).astype(np.half).astype(np.cfloat).astype(np.half).item()
    return prob


def compare_scalar(rst_lst, prob):
    rst_np = np.array(rst_lst)
    prob_np = np.array(prob)
    mask = rst_np <= prob_np
    rst_np[mask] = 1
    rst_np[~mask] = 0
    return rst_np


def binary_array_to_uint8(binary_array):
    binary_array = np.array(binary_array, dtype=np.uint8)
    padding = 8 - (len(binary_array) % 8)
    if padding != 8:
        binary_array = np.pad(
            binary_array, (0, padding), mode="constant", constant_values=0
        )
    uint8_values = np.packbits(binary_array, bitorder="little")
    return uint8_values


def uniform_pt(philox_random, prob, dtype):
    uniform_out = curand_uniform(philox_random)
    prob = update_prob_type(prob, dtype)
    return uniform_out, prob


def GetVectorSize(eleCount, T_size):
    vecSize = 8
    if eleCount % 2 != 0:
        return 1
    optimalVecSize = 16 // T_size
    vecSize = min(vecSize, optimalVecSize)
    while vecSize > 1:
        canVectorize = (eleCount % vecSize) == 0
        if not canVectorize:
            vecSize = vecSize // 2
        else:
            break
    return vecSize


def drop_out_v3_compute(x_in, maskout, prob, seed, offset, rounds=10):
    count = reduce(lambda x, y: x * y, x_in.shape)
    if prob == 0:
        y_out = torch.zeros(x_in.shape, dtype=x_in.dtype, device=x_in.device)
        binary_array = np.zeros(maskout.shape, dtype=np.uint8)
        return y_out, binary_array

    if prob == 1.0:
        binary_array = np.full(maskout.shape, np.uint8(-1), dtype=np.uint8)
        return x_in, binary_array

    blockSize = 256
    maxThreadsPerMultiProcessor = 2048
    blocksPerSM = maxThreadsPerMultiProcessor // blockSize
    multiProcessorCount = 78
    grid = (count + blockSize - 1) // blockSize
    grid = min(multiProcessorCount * blocksPerSM, grid)
    totalThreads = grid * blockSize

    T_size = 4 if x_in.dtype == torch.float32 else 2
    vecSize = GetVectorSize(count, T_size)

    mask_out = np.zeros(count, dtype=bool)
    y_out = copy.deepcopy(x_in.flatten())

    key = [seed & MASK_32, (seed >> 32) & MASK_32]

    if vecSize == 1:
        for idx in range(0, count, vecSize):
            threadIdx = idx % totalThreads
            repeatCount = idx // totalThreads
            (key_g, counter) = gen_key_and_counter(
                threadIdx, seed, offset // 4 + repeatCount // 4
            )
            philox_random = philox(rounds, counter, key_g, 4)
            (uniform_out, prob) = uniform_pt(philox_random, prob, "float")
            mask = compare_scalar(uniform_out, prob)
            mask_out[idx] = mask[repeatCount % 4]
    else:
        fixOffset = vecSize
        if vecSize == 2:
            fixOffset = 4

        num_vec = count // vecSize
        rand_per_vec = (vecSize + 3) // 4
        batch = 8192

        for batch_start in range(0, num_vec, batch):
            batch_end = min(batch_start + batch, num_vec)
            batch_size = batch_end - batch_start

            vecIndx_arr = np.arange(batch_start, batch_end, dtype=np.uint64)
            threadIdx_arr = vecIndx_arr % totalThreads
            repeatCount_arr = vecIndx_arr // totalThreads

            counters = np.zeros((batch_size * rand_per_vec, 4), dtype=np.uint64)
            for r in range(rand_per_vec):
                eff_vals = [
                    int(offset // 4 + int(rc) * (fixOffset // 4) + r)
                    & 0xFFFFFFFFFFFFFFFF
                    for rc in repeatCount_arr
                ]
                eff_offset_arr = np.array(eff_vals, dtype=np.uint64)
                sl = slice(r * batch_size, (r + 1) * batch_size)
                counters[sl, 0] = eff_offset_arr & MASK64
                counters[sl, 1] = (eff_offset_arr >> np.uint64(32)) & MASK64
                counters[sl, 2] = threadIdx_arr & MASK64
                counters[sl, 3] = (threadIdx_arr >> np.uint64(32)) & MASK64

            result = philox_10_rounds_vec(counters, key)
            u32 = result.astype(np.uint32)
            uniform = (
                u32.astype(np.float64) * np.float64(2.0**-32) + np.float64(2.0**-33)
            ).astype(np.float32)
            mask = uniform <= prob

            if rand_per_vec == 1:
                mask_flat = mask[:, :vecSize].flatten()
            else:
                mask_flat = np.empty(batch_size * vecSize, dtype=bool)
                for r in range(rand_per_vec):
                    num = min(vecSize - r * 4, 4)
                    dst_start = r * 4
                    if num > 0:
                        for i in range(batch_size):
                            mask_flat[
                                i * vecSize + dst_start : i * vecSize + dst_start + num
                            ] = mask[r * batch_size + i, 0:num]
                mask_flat = mask_flat[: batch_size * vecSize]

            idx_start = batch_start * vecSize
            idx_end = batch_end * vecSize
            mask_out[idx_start:idx_end] = mask_flat[: idx_end - idx_start]

    mask_bool = mask_out[:count]
    y_out[mask_bool] = y_out[mask_bool] * (1 / prob)
    y_out[~mask_bool] = y_out[~mask_bool] * 0
    mask_uint8 = mask_to_align16(binary_array_to_uint8(mask_out))

    return y_out.reshape(x_in.shape), mask_uint8


def compute_data(
    input,
    optionalNoiseShapeTensor,
    p,
    seed,
    offset,
    out,
    maskout,
):
    x_tensor = _as_torch(input).detach().cpu()
    x_shape = x_tensor.shape
    if not x_tensor.dim():
        x_tensor = torch.tensor([x_tensor])
    tol = reduce(lambda x, y: x * y, x_shape)
    if tol == 0:
        return [torch.empty(x_shape), torch.empty(x_shape)]
    p = 1 - p

    dst, mask = drop_out_v3_compute(x_tensor, maskout, p, seed, offset)
    out_dtype = out.dtype
    dst_torch = dst.to(out_dtype).reshape(x_shape)
    mask_torch = torch.from_numpy(mask)
    out = dst_torch
    maskout = mask_torch
    return [out, maskout]


class ThirdPartyImplTensor:
    def __init__(self, *args, **kwargs):
        (
            input,
            optionalNoiseShapeTensor,
            p,
            seedTensor,
            offsetTensor,
            offset,
            out,
            maskout,
        ) = (list(args) + [None] * 8)[:8]
        self.offset = offsetTensor[0].item() + offset
        self.p = p
        self.seed = seedTensor[0].item()
        self.input = input.clone()
        self.outDtype = out.dtype

    def __call__(self, **kwargs):
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        torch.cuda.manual_seed(self.seed)
        default_gen = torch.cuda.default_generators[0]
        if hasattr(default_gen, "set_offset"):
            default_gen.set_offset(self.offset)

        dst = torch.dropout(self.input, p=self.p, train=False)
        dst = dst.to(self.outDtype)
        mask = torch.tensor(0)
        return [dst, mask]


class ThirdPartyImpl:
    def __init__(self, *args, **kwargs):
        input_t, optionalNoiseShapeTensor, p, seed, offset, out, maskout = (
            list(args) + [None] * 8
        )[:8]
        self.offset = offset
        self.p = p
        self.seed = seed
        self.input = input_t.clone()
        self.outDtype = out.dtype

    def __call__(self, **kwargs):
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        torch.cuda.manual_seed(self.seed)
        default_gen = torch.cuda.default_generators[0]
        if hasattr(default_gen, "set_offset"):
            default_gen.set_offset(self.offset)

        dst = torch.dropout(self.input, p=self.p, train=False)
        dst = dst.to(self.outDtype)
        mask = torch.tensor(0)
        return [dst, mask]


class aclnnDropoutV3Spec:
    """ACLNN 流程 — golden / third_party 均收到 torch.Tensor（已在设备上）"""

    def golden(
        input,
        optionalNoiseShapeTensor,
        p,
        seed,
        offset,
        out,
        maskout,
        **kwargs,
    ):
        return compute_data(
            input, optionalNoiseShapeTensor, p, seed, offset, out, maskout
        )

    third_party = {"torch": ThirdPartyImpl}
    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }


class aclnnDropoutV3TensorSpec:
    """ACLNN 流程 — golden / third_party 均收到 torch.Tensor（已在设备上）"""

    def golden(
        input,
        optionalNoiseShapeTensor,
        p,
        seedTensor,
        offsetTensor,
        offset,
        out,
        maskout,
        **kwargs,
    ):
        seed = int(_as_torch(seedTensor).reshape(-1)[0].item())
        offset2 = int(_as_torch(offsetTensor).reshape(-1)[0].item())
        realOffset = offset + offset2
        return compute_data(
            input, optionalNoiseShapeTensor, p, seed, realOffset, out, maskout
        )

    third_party = {"torch": ThirdPartyImplTensor}
    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }
