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

__golden__ = {"kernel": {"get_shape": "get_shape_golden"}}

__spec__ = {"get_shape": "GetShapeTestSpec"}

X_DESCRIPTOR_SIZE = 128


def get_shape_golden(*inputs, **kwargs):
    shapes = []
    for inp in inputs:
        if inp.shape == (X_DESCRIPTOR_SIZE,) and np.issubdtype(inp.dtype, np.integer):
            dim_num = int(inp[3])
            for j in range(dim_num):
                shapes.append(int(inp[4 + j]))
        else:
            shapes.extend(inp.shape)
    return np.array(shapes, dtype=np.int32)


class GetShapeTorchImpl:
    def __call__(self, *inputs, input_formats=None, **kwargs):
        import torch

        if len(inputs) == 1 and isinstance(inputs[0], (list, tuple)):
            inputs = inputs[0]
        parts = []
        for inp in inputs:
            inp = inp if isinstance(inp, torch.Tensor) else torch.as_tensor(inp)
            if (
                tuple(inp.shape) == (X_DESCRIPTOR_SIZE,)
                and not inp.is_floating_point()
                and not inp.is_complex()
                and inp.dtype != torch.bool
            ):
                rank = int(inp[3].item())
                if not 0 <= rank <= X_DESCRIPTOR_SIZE - 4:
                    raise ValueError("Invalid shape descriptor rank")
                parts.append(inp[4 : 4 + rank].to(dtype=torch.int32))
            else:
                parts.append(
                    torch._shape_as_tensor(inp).to(dtype=torch.int32, device=inp.device)
                )
        return torch.cat(parts) if parts else torch.empty(0, dtype=torch.int32)


class GetShapeTFImpl:
    def __call__(self, *inputs, input_formats=None, **kwargs):
        import tensorflow as tf
        import torch

        if len(inputs) == 1 and isinstance(inputs[0], (list, tuple)):
            inputs = inputs[0]
        arrays = [
            inp.detach().cpu().numpy()
            if isinstance(inp, torch.Tensor)
            else np.asarray(inp)
            for inp in inputs
        ]

        def build():
            parts = []
            for arr in arrays:
                tensor = tf.convert_to_tensor(arr)
                if arr.shape == (X_DESCRIPTOR_SIZE,) and np.issubdtype(
                    arr.dtype, np.integer
                ):
                    rank = int(arr[3])
                    if not 0 <= rank <= X_DESCRIPTOR_SIZE - 4:
                        raise ValueError("Invalid shape descriptor rank")
                    parts.append(tf.cast(tf.slice(tensor, [4], [rank]), tf.int32))
                else:
                    parts.append(tf.shape(tensor, out_type=tf.int32))
            return (
                tf.concat(parts, axis=0) if parts else tf.constant([], dtype=tf.int32)
            )

        if tf.executing_eagerly():
            out_np = build().numpy()
        else:
            graph = tf.Graph()
            with graph.as_default():
                out_g = build()
            with tf.compat.v1.Session(graph=graph) as sess:
                out_np = sess.run(out_g)
        return np.asarray(out_np)


class GetShapeTestSpec:
    def golden(*inputs, **kwargs):
        if len(inputs) == 1 and isinstance(inputs[0], (list, tuple)):
            inputs = inputs[0]
        return [get_shape_golden(*inputs, **kwargs)]

    third_party = {"torch": GetShapeTorchImpl, "tf": GetShapeTFImpl}
    tolerance = {"int32": {"standard": "binary_equal"}}
