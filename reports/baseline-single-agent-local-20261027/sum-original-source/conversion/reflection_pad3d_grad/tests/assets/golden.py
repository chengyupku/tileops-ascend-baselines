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

__golden__ = {"kernel": {"reflection_pad3d_grad": "reflection_pad3d_grad_golden"}}

__spec__ = {"reflection_pad3d_grad": "ReflectionPad3dGradTestSpec"}


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


def _cal_out_shape(in_shape, paddings):
    out_shape = []
    offset = len(in_shape) - len(paddings)
    for i in range(len(in_shape)):
        if i < len(in_shape) - len(paddings):
            out_shape.append(in_shape[i])
        else:
            out_shape.append(
                in_shape[i] - paddings[i - offset][0] - paddings[i - offset][1]
            )
    return out_shape


def reflection_pad3d_grad_golden(x, paddings, **kwargs):
    """
    Kernel golden for reflection_pad3d_grad.
    All the parameters follow @reflection_pad3d_grad_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    """
    import torch

    grad_output_np = x
    paddings_arr = np.array(paddings).astype(np.int64).reshape(-1, 2)
    y_shape = _cal_out_shape(grad_output_np.shape, paddings_arr.tolist())

    origin_dtype = grad_output_np.dtype.name
    if origin_dtype == "bfloat16":
        grad_output = _numpy_to_torch_tensor(grad_output_np).to(torch.float32)
    else:
        grad_output = _numpy_to_torch_tensor(grad_output_np)

    pad_pairs = paddings_arr
    while len(pad_pairs) > 3 and not pad_pairs[0].any():
        pad_pairs = pad_pairs[1:]
    y = torch.zeros(y_shape, dtype=grad_output.dtype)
    y.requires_grad_(True)
    padded = torch.nn.functional.pad(
        y, [p for pair in reversed(pad_pairs) for p in pair], "reflect"
    )

    (grad_output * padded).sum().backward()
    golden = y.grad

    if origin_dtype == "bfloat16":
        golden = _torch_to_numpy_tensor(golden.to(torch.bfloat16))
    else:
        golden = golden.numpy()
    return golden


class ReflectionPad3dGradImpl:
    def __call__(self, x, paddings):
        import torch

        paddings_arr = np.asarray(
            torch.as_tensor(paddings).detach().cpu().tolist(), dtype=np.int64
        ).reshape(-1, 2)
        y_shape = _cal_out_shape(x.shape, paddings_arr.tolist())
        pad_pairs = paddings_arr
        while len(pad_pairs) > 3 and not pad_pairs[0].any():
            pad_pairs = pad_pairs[1:]
        pad_list = [p for pair in reversed(pad_pairs) for p in pair]
        return torch.ops.aten.reflection_pad3d_backward(
            x, torch.empty(y_shape, dtype=x.dtype, device=x.device), pad_list
        )


class ReflectionPad3dGradTestSpec:
    def golden(x, paddings, **kwargs):
        return [reflection_pad3d_grad_golden(x, paddings, **kwargs)]

    third_party = {"torch": ReflectionPad3dGradImpl}
