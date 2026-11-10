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

import typing

import numpy as np
import torch

try:
    import ml_dtypes as _ml_dtypes
except ImportError:
    _ml_dtypes = None

__spec__ = {
    "bias": "BiasKernelSpec",
}


_TOL = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
}

_TORCH_DTYPE = {
    "float64": torch.float64,
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

_NP_DTYPE = {
    "float64": np.float64,
    "float32": np.float32,
    "float16": np.float16,
    "bfloat16": np.float32,
}


def _attr(kwargs, name, default):
    v = kwargs.get(name, default)
    if isinstance(v, str):
        s = v.strip()
        if isinstance(default, bool):
            sl = s.lower()
            if sl in ("true", "false", "yes", "no", "1", "0"):
                return sl in ("true", "yes", "1")
            return default
        try:
            return type(default)(s)
        except (TypeError, ValueError):
            return default
    return v


def _np_to_torch(arr):
    arr = np.ascontiguousarray(arr)
    if arr.dtype == np.uint16:
        return torch.from_numpy(arr.copy()).view(torch.bfloat16)
    if _ml_dtypes is not None and arr.dtype == _ml_dtypes.bfloat16:
        return torch.from_numpy(arr.view(np.uint16).copy()).view(torch.bfloat16)
    return torch.from_numpy(arr)


def _bias_reshape_shape(x, bias, axis, num_axes, bias_from_blob):
    rank = x.ndim
    if axis < 0:
        axis += rank
    if bias_from_blob:
        if num_axes == -1:
            return [1] * axis + list(bias.shape)
        if num_axes == 0:
            return [1] * rank
        return [1] * axis + list(bias.shape) + [1] * (rank - axis - num_axes)
    if bias.ndim == 1 and bias.shape[0] == 1:
        return [1] * rank
    return [1] * axis + list(bias.shape) + [1] * (rank - axis - bias.ndim)


def _compute(*tensors, **kwargs):
    """全程 torch.Tensor 进、torch.Tensor 出。返回 list[Tensor]。

    R3: golden 使用 torch.add，非 numpy 纯公式。
    """
    x, bias = tensors[0], tensors[1]
    axis = _attr(kwargs, "axis", 1)
    num_axes = _attr(kwargs, "num_axes", 1)
    bias_from_blob = _attr(kwargs, "bias_from_blob", True)
    reshape_shape = _bias_reshape_shape(x, bias, axis, num_axes, bias_from_blob)
    bias_reshaped = bias.reshape(reshape_shape)
    if x.dtype in (torch.float16, torch.bfloat16):
        return [torch.add(x.float(), bias_reshaped.float())]
    return [torch.add(x, bias_reshaped)]


def _torch_to_np(t, np_dt):
    if t.dtype == torch.bfloat16:
        t = t.float()
    arr = t.detach().cpu().numpy()
    return arr if arr.dtype == np_dt else arr.astype(np_dt)


class _Compose:
    """torch 原生算子拼出等价语义。"""

    def __init__(self, **kwargs):
        self.axis = _attr(kwargs, "axis", 1)
        self.num_axes = _attr(kwargs, "num_axes", 1)
        self.bias_from_blob = _attr(kwargs, "bias_from_blob", True)

    def __call__(self, x, bias, **kwargs):
        """Explicit inputs keep remote name-based binding from treating attrs as tensors."""
        reshape_shape = _bias_reshape_shape(
            x, bias, self.axis, self.num_axes, self.bias_from_blob
        )
        bias_reshaped = bias.reshape(reshape_shape)
        if x.dtype in (torch.float16, torch.bfloat16):
            return [torch.add(x.float(), bias_reshaped.float()).to(x.dtype)]
        return [torch.add(x, bias_reshaped)]


class BiasKernelSpec:
    """kernel + geir 共用。golden 收 numpy.ndarray，返 numpy.ndarray。"""

    def golden(*inputs, **kwargs):
        tensors = []
        for arr in inputs:
            if isinstance(arr, np.ndarray):
                tensors.append(_np_to_torch(arr))
            else:
                tensors.append(arr)
        outs = _compute(*tensors, **kwargs)
        od = kwargs.get("output_dtypes") or []
        od = [d[0] if isinstance(d, (list, tuple)) else str(d) for d in od]
        result = []
        for i, o in enumerate(outs):
            if i < len(od):
                target = _TORCH_DTYPE.get(od[i])
                if target is not None:
                    o = o.to(target)
            np_dt = _NP_DTYPE.get(od[i] if i < len(od) else "", np.float32)
            result.append(_torch_to_np(o, np_dt))
        return result

    third_party: typing.ClassVar[dict] = {"torch": _Compose}
    tolerance: typing.ClassVar[dict] = _TOL
