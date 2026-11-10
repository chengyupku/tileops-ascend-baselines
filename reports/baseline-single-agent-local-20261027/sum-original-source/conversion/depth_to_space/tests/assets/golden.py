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

__golden__ = {"kernel": {"depth_to_space": "depth_to_space"}}


__spec__ = {"depth_to_space": "DepthToSpaceTestSpec"}


def depth_to_space(x, *, block_size, mode, data_format, **kwargs):
    """
    Kernel golden for depth_to_space.
    All the parameters follow @depth_to_space_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    """
    shapes = x.shape
    if data_format == "NCHW":
        n, c, h, w = shapes
        output_shapes = [n, c // (block_size**2), h * block_size, w * block_size]
        if mode == "CRD":
            input_shapes = [n, c // (block_size**2), block_size, block_size, h, w]
            perm = [0, 1, 4, 2, 5, 3]
        else:
            input_shapes = [n, block_size, block_size, c // (block_size**2), h, w]
            perm = [0, 3, 4, 1, 5, 2]
    else:
        n, h, w, c = shapes
        output_shapes = [n, h * block_size, w * block_size, c // (block_size**2)]
        if mode == "CRD":
            input_shapes = [n, h, w, c // (block_size**2), block_size, block_size]
            perm = [0, 1, 4, 2, 5, 3]
        else:
            input_shapes = [n, h, w, block_size, block_size, c // (block_size**2)]
            perm = [0, 1, 3, 2, 4, 5]
    tmp = x.reshape(input_shapes)
    tmp = np.transpose(tmp, perm)
    return tmp.reshape(output_shapes)


class DepthToSpaceTorchImpl:
    def __call__(self, x, block_size, mode="DCR", data_format="NHWC"):
        import torch

        if mode != "CRD":
            raise NotImplementedError("PyTorch pixel_shuffle only supports CRD mode")
        if data_format not in ("NCHW", "NHWC"):
            raise NotImplementedError(f"Unsupported data format: {data_format}")
        if not isinstance(x, torch.Tensor):
            x_np = np.asarray(x)
            x = (
                torch.from_numpy(x_np.view(np.int16)).view(torch.bfloat16)
                if x_np.dtype.name == "bfloat16"
                else torch.from_numpy(x_np)
            )
        if data_format == "NHWC":
            x = x.permute(0, 3, 1, 2)
        out = torch.nn.functional.pixel_shuffle(x, int(block_size))
        return out.permute(0, 2, 3, 1).contiguous() if data_format == "NHWC" else out


class DepthToSpaceTFImpl:
    def __call__(self, x, block_size, mode="DCR", data_format="NHWC"):
        import tensorflow as tf
        import torch

        if mode != "DCR":
            raise NotImplementedError("TensorFlow DepthToSpace only supports DCR mode")
        if data_format not in ("NCHW", "NHWC"):
            raise NotImplementedError(f"Unsupported data format: {data_format}")
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu()
            if x.dtype == torch.bfloat16:
                from ml_dtypes import bfloat16

                x_np = x.view(torch.int16).numpy().view(bfloat16)
            else:
                x_np = x.numpy()
        else:
            x_np = np.asarray(x)

        def build():
            x_tf = tf.convert_to_tensor(x_np)
            # CPU DepthToSpace 使用 NHWC，NCHW 只做布局转换。
            if data_format == "NCHW":
                x_tf = tf.transpose(x_tf, [0, 2, 3, 1])
            out = tf.raw_ops.DepthToSpace(
                input=x_tf, block_size=int(block_size), data_format="NHWC"
            )
            return tf.transpose(out, [0, 3, 1, 2]) if data_format == "NCHW" else out

        if tf.executing_eagerly():
            out_np = build().numpy()
        else:
            graph = tf.Graph()
            with graph.as_default():
                out_g = build()
            with tf.compat.v1.Session(graph=graph) as sess:
                out_np = sess.run(out_g)
        return np.asarray(out_np)


class DepthToSpaceTestSpec:
    def golden(x, block_size, mode="DCR", data_format="NHWC", **kwargs):
        return [
            depth_to_space(x, block_size=block_size, mode=mode, data_format=data_format)
        ]

    third_party = {"torch": DepthToSpaceTorchImpl, "tf": DepthToSpaceTFImpl}

    tolerance = {
        "float32": {"standard": "binary_equal"},
        "float16": {"standard": "binary_equal"},
        "bfloat16": {"standard": "binary_equal"},
    }
