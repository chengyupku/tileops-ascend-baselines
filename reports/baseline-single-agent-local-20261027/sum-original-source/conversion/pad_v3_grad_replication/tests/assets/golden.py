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


__golden__ = {"kernel": {"pad_v3_grad_replication": "pad_v3_grad_replication_golden"}}

__spec__ = {"pad_v3_grad_replication": "PadV3GradReplicationTestSpec"}


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


def _numpy_pad_v3_grad_edge(grad_output, in_shape, pad_per_dim):
    dim = grad_output.ndim

    true_dtype = grad_output.dtype
    grad_input = np.zeros(in_shape, dtype=np.float32)
    grad_output = np.array(grad_output, dtype=np.float32)

    out_shape = grad_output.shape
    for out_idx in np.ndindex(out_shape):
        in_idx = []
        pad_offset = dim - len(pad_per_dim)
        for i in range(dim):
            left, right = pad_per_dim[i - pad_offset] if i >= pad_offset else (0, 0)
            out_len = out_shape[i]
            in_len = in_shape[i]
            o = out_idx[i]

            left_pad = max(0, left)
            right_pad = max(0, right)

            if left_pad > 0 and o < left_pad:
                i_in = 0
            elif right_pad > 0 and o >= out_len - right_pad:
                i_in = in_len - 1
            else:
                i_in = o - left

            in_idx.append(i_in)

        in_idx = tuple(in_idx)
        grad_input[in_idx] += grad_output[out_idx]
    grad_input = np.array(grad_input, dtype=true_dtype)
    return grad_input


def pad_v3_grad_replication_golden(x, paddings, **kwargs):
    """
    Kernel golden for pad_v3_grad_replication.
    All the parameters follow @pad_v3_grad_replication_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    """
    grad_output = x

    paddings_arr = np.array(paddings).astype(np.int64)
    if paddings_arr.ndim == 1:
        paddings_arr = paddings_arr.reshape(-1, 2)
    pad_shape = paddings_arr.tolist()

    y_shape = _cal_out_shape(grad_output.shape, pad_shape)

    grad = _numpy_pad_v3_grad_edge(grad_output, y_shape, pad_shape)
    return grad


class PadV3GradReplicationImpl:
    def __call__(self, x, paddings, **kwargs):
        import torch

        grad_output = x if isinstance(x, torch.Tensor) else torch.from_numpy(x)
        paddings_arr = np.array(
            paddings
            if not isinstance(paddings, torch.Tensor)
            else paddings.detach().cpu().numpy()
        )
        paddings_arr = paddings_arr.astype(np.int64).reshape(-1, 2)
        y_shape = _cal_out_shape(list(grad_output.shape), paddings_arr.tolist())
        while len(paddings_arr) and not paddings_arr[0].any():
            paddings_arr = paddings_arr[1:]
        if not len(paddings_arr):
            return grad_output.clone()
        count = len(paddings_arr)
        if count not in (1, 2, 3) or grad_output.dim() not in (count + 1, count + 2):
            raise NotImplementedError(
                "PyTorch replication padding backward requires trailing 1-3 "
                "dimensions with one or two leading dimensions"
            )
        pad_list = [p for pair in reversed(paddings_arr) for p in pair]
        aten = torch.ops.aten
        fn = {
            1: aten.replication_pad1d_backward,
            2: aten.replication_pad2d_backward,
            3: aten.replication_pad3d_backward,
        }[len(paddings_arr)]
        return fn(
            grad_output,
            torch.empty(y_shape, dtype=grad_output.dtype, device=grad_output.device),
            pad_list,
        )


class PadV3GradReplicationTestSpec:
    def golden(x, paddings, **kwargs):
        return [pad_v3_grad_replication_golden(x, paddings, **kwargs)]

    third_party = {"torch": PadV3GradReplicationImpl}
