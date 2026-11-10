#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

__spec__ = {
    "slice_with_axes": "SliceWithAxesKernelSpec",
}

import numpy as np
import torch
import ml_dtypes


def _to_torch(arr):
    arr = np.ascontiguousarray(arr)
    if arr.dtype == ml_dtypes.bfloat16:
        return torch.from_numpy(arr.view(np.int16)).view(torch.bfloat16)
    return torch.from_numpy(arr)


def _to_np(t):
    t = t.contiguous()
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16).numpy().view(ml_dtypes.bfloat16)
    return t.numpy()


def _narrow_axes(x, axes, offsets, sizes):
    # 沿 axes 各轴 narrow：y[..., axis_k, ...] = x[..., offsets[k] : offsets[k] + size[k], ...]
    # size[k] == -1 表示从 offsets[k] 切到该轴末尾；golden 与 ThirdParty 共用本函数
    for ax, off, sz in zip(axes, offsets, sizes):
        off, sz = int(off), int(sz)
        sz = x.shape[int(ax)] - off if sz == -1 else sz
        x = x.narrow(int(ax), off, sz)
    return x


class SliceWithAxesThirdParty:
    """Timed torch benchmark: 多轴切片用 narrow 链实现，与 golden 共用 _narrow_axes。

    算子语义与 TF Slice 的按轴稀疏形式一致。
    """

    def __init__(self, *, axes, **kwargs):
        self.axes = [int(a) for a in axes]

    def __call__(self, x, offsets, size, **kwargs):
        return [
            _narrow_axes(x, self.axes, offsets.tolist(), size.tolist()).contiguous()
        ]


_BINARY_TOLERANCE = {
    "float16": {"standard": "binary_equal"},
    "float32": {"standard": "binary_equal"},
    "float64": {"standard": "binary_equal"},
    "bfloat16": {"standard": "binary_equal"},
    "int8": {"standard": "binary_equal"},
    "int16": {"standard": "binary_equal"},
    "int32": {"standard": "binary_equal"},
    "int64": {"standard": "binary_equal"},
    "uint8": {"standard": "binary_equal"},
    "uint16": {"standard": "binary_equal"},
    "uint32": {"standard": "binary_equal"},
    "uint64": {"standard": "binary_equal"},
    "bool": {"standard": "binary_equal"},
}


class SliceWithAxesKernelSpec:
    def golden(x, offsets, size, axes, **kwargs):
        y = _narrow_axes(_to_torch(x), axes, offsets.tolist(), size.tolist())
        return [_to_np(y)]

    third_party = {"torch": SliceWithAxesThirdParty}
    tolerance = _BINARY_TOLERANCE
