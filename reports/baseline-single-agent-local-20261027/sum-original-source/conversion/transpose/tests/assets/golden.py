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

"""Golden for Transpose operator (Kernel and aclnn Permute)."""

__spec__ = {
    "transpose": "TransposeKernelSpec",
    "aclnnPermute": "AclnnPermuteSpec",
}

import numpy as np
import torch


def _parse_perm(perm):
    if hasattr(perm, "tolist"):
        perm = perm.tolist()
    if isinstance(perm, int):
        return [perm]
    return list(perm)


class TransposeImpl:
    def __call__(self, x, perm, **kwargs):
        if not isinstance(x, torch.Tensor):
            x_np = np.asarray(x)
            x = (
                torch.from_numpy(x_np.view(np.int16)).view(torch.bfloat16)
                if x_np.dtype.name == "bfloat16"
                else torch.from_numpy(x_np)
            )
        return torch.permute(x, _parse_perm(perm)).contiguous()


class AclnnPermuteImpl:
    def __call__(self, *inputs, dims=0, out=None, input_formats=None, **kwargs):
        # ACLNN names the input tensor "self"; the remote binder forwards it
        # through *inputs to avoid colliding with the Python instance parameter.
        if len(inputs) != 1:
            raise ValueError("aclnnPermute expects exactly one input tensor")
        return TransposeImpl()(inputs[0], dims)


class TransposeTFImpl:
    def __call__(self, x, perm, **kwargs):
        import tensorflow as tf

        if isinstance(x, torch.Tensor):
            x = x.detach().cpu()
            if x.dtype == torch.bfloat16:
                from ml_dtypes import bfloat16

                x_np = x.view(torch.int16).numpy().view(bfloat16)
            else:
                x_np = x.numpy()
        else:
            x_np = np.asarray(x)
        perm = [axis + x_np.ndim if axis < 0 else axis for axis in _parse_perm(perm)]
        if tf.executing_eagerly():
            out_np = tf.transpose(tf.convert_to_tensor(x_np), perm=perm).numpy()
        else:
            graph = tf.Graph()
            with graph.as_default():
                out_g = tf.transpose(tf.convert_to_tensor(x_np), perm=perm)
            with tf.compat.v1.Session(graph=graph) as sess:
                out_np = sess.run(out_g)
        return np.asarray(out_np)


_TOLERANCE = {
    "float32": {"standard": "binary_equal"},
    "float16": {"standard": "binary_equal"},
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
    "hifloat8": {"standard": "binary_equal"},
    "float8_e5m2": {"standard": "binary_equal"},
    "float8_e4m3fn": {"standard": "binary_equal"},
}


class TransposeKernelSpec:
    def golden(x, perm, **kwargs):
        perm_val = _parse_perm(perm)
        return [np.transpose(x, perm_val)]

    third_party = {"torch": TransposeImpl, "tf": TransposeTFImpl}
    tolerance = _TOLERANCE


class AclnnPermuteSpec:
    def golden(self, dims=0, out=None, **kwargs):
        if hasattr(dims, "tolist"):
            dims = dims.tolist()
        elif isinstance(dims, int):
            dims = [dims]
        return [torch.permute(self, dims).contiguous()]

    third_party = {"torch": AclnnPermuteImpl}
    tolerance = _TOLERANCE
