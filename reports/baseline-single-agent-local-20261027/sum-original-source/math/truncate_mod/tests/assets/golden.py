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

import numpy as np


__golden__ = {"kernel": {"truncate_mod": "truncate_mod_golden"}}


__spec__ = {"truncate_mod": "TruncateModTestSpec"}


def _broadcast_to_maxshape(shapes):
    """
    produce broadcast shape
    for example:
        input: shape is [[2, 3], [3, 2, 1], [3, 1, 3]]
        output: [1, 2, 3], [3, 2, 1], [3, 1, 3], [3, 2, 3]
    """

    def _max(_shape):
        no_one_shape = [s for s in _shape if s != 1]
        if len(no_one_shape) == 0:
            max_value = 1
        else:
            max_value = no_one_shape[0]
        return max_value

    max_dim_length = max(len(list(shape)) for shape in shapes)
    input_shapes = []
    for shape in shapes:
        input_shapes.append(
            [1 for _ in range(max_dim_length - len(shape))] + list(shape)
        )
    input_shapes = list(map(list, zip(*input_shapes)))
    max_shape = [_max(shape) for shape in input_shapes]
    input_shapes = list(map(list, zip(*input_shapes)))
    return (*input_shapes, max_shape)


def _numpy_bfloat16():
    try:
        from ml_dtypes import bfloat16
    except ModuleNotFoundError:
        try:
            import tensorflow

            bfloat16 = tensorflow.bfloat16.as_numpy_dtype
        except ModuleNotFoundError:
            raise RuntimeError(
                "ml-dtypes or tensorflow is needed to support bfloat16 dtype!!! "
                "Please install with `pip3 install ml-dtypes` or `pip3 install tensorflow`"
            )
    return bfloat16


def _numpy_to_torch_tensor(np_array):
    import torch

    if np_array is None:
        return None
    np_dtype = np_array.dtype.name
    if "bfloat16" in np_dtype:
        np_int16 = np_array.view(dtype=np.int16)
        t_int16 = torch.from_numpy(np_int16)
        return t_int16.view(torch.bfloat16)
    else:
        return torch.from_numpy(np_array)


def _torch_to_numpy_tensor(torch_tensor):
    import torch

    if torch_tensor is None:
        return None
    if not isinstance(torch_tensor, torch.Tensor):
        raise RuntimeError(f"Only support torch.Tensor. But got {type(torch_tensor)}")
    torch_dtype = torch_tensor.dtype
    if torch_dtype == torch.bfloat16:
        t_int16 = torch_tensor.view(torch.int16)
        np_int16 = t_int16.numpy()
        return np_int16.view(dtype=_numpy_bfloat16())
    else:
        return torch_tensor.numpy()


def _safe_integer_divisor(x1, x2):
    import torch

    if x1.dtype == x2.dtype and x1.dtype in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        overflow = (x1 == torch.iinfo(x1.dtype).min) & (x2 == -1)
        return torch.where(overflow, torch.ones_like(x2), x2)
    return x2


def truncate_mod_golden(x1, x2, **kwargs):
    """
    Kernel golden for truncate_mod.
    All the parameters follow @truncate_mod_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    """
    import torch

    output_dtype = kwargs.get("output_dtypes", [None])[0]
    if output_dtype is None:
        output_dtype = str(x1.dtype)

    type_int = [torch.int8, torch.int16, torch.int32, torch.int64]
    type_uint = [torch.uint8, torch.uint16, torch.uint32, torch.uint64]
    type_float = [torch.float16, torch.bfloat16, torch.float, torch.float64]

    x1 = _numpy_to_torch_tensor(x1.copy())
    x2 = _numpy_to_torch_tensor(x2.copy())
    x1_b, x2_b = torch.broadcast_tensors(x1, x2)
    zero_mask = x2_b == 0
    x2_safe = torch.where(zero_mask, torch.ones_like(x2_b), x2_b)
    res = torch.fmod(x1_b, _safe_integer_divisor(x1_b, x2_safe))

    if res.dtype in type_int:
        res = torch.where(zero_mask, torch.full_like(res, -1), res)
    elif res.dtype in type_uint:
        res = torch.where(zero_mask, torch.full_like(res, 255), res)
    elif res.dtype in type_float:
        res = torch.where(zero_mask, torch.full_like(res, torch.nan), res)

    res_np = _torch_to_numpy_tensor(res)
    return res_np.astype(output_dtype, copy=False)


class TruncateModImpl:
    def __call__(self, x1, x2, **kwargs):
        import torch

        x1 = torch.as_tensor(x1)
        x2 = torch.as_tensor(x2)
        output_dtype = str(
            kwargs.get("output_dtypes", [None])[0] or x1.dtype
        ).removeprefix("torch.")
        return self._core(x1, x2, output_dtype)

    def _core(self, x1, x2, output_dtype):
        import torch

        x1_b, x2_b = torch.broadcast_tensors(x1, x2)
        zero_mask = x2_b == 0
        x2_safe = torch.where(zero_mask, torch.ones_like(x2_b), x2_b)
        res = torch.fmod(x1_b, _safe_integer_divisor(x1_b, x2_safe))

        if res.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
            res = torch.where(zero_mask, torch.full_like(res, -1), res)
        elif res.dtype == torch.uint8:
            res = torch.where(zero_mask, torch.full_like(res, 255), res)
        elif res.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            res = torch.where(zero_mask, torch.full_like(res, torch.nan), res)
        return (
            res.to(getattr(torch, output_dtype))
            if res.dtype != getattr(torch, output_dtype)
            else res
        )


class TruncateModTestSpec:
    def golden(x1, x2, **kwargs):
        return [truncate_mod_golden(x1, x2, **kwargs)]

    third_party = {"torch": TruncateModImpl}

    tolerance = {
        "float32": {"standard": "mix_tolerance"},
        "float16": {"standard": "mix_tolerance"},
        "bfloat16": {"standard": "mix_tolerance"},
        "int8": {"standard": "binary_equal"},
        "int16": {"standard": "binary_equal"},
        "int32": {"standard": "binary_equal"},
        "int64": {"standard": "binary_equal"},
        "uint8": {"standard": "binary_equal"},
    }
