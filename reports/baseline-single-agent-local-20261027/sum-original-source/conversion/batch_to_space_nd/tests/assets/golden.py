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


__golden__ = {"kernel": {"batch_to_space_nd": "batch_to_space_nd_golden"}}


__spec__ = {"batch_to_space_nd": "BatchToSpaceNdTestSpec"}


def _ceil_div(x, y):
    return (x + y - 1) // y


def shape_6d_2_5d(tensor):
    n, di, c1, hi, wi, c0 = tensor.shape
    tmp_tensor = tensor.reshape(n, di, c1, hi, wi, c0)
    tmp_tensor = np.transpose(tmp_tensor, axes=(0, 1, 3, 4, 2, 5))
    new_tensor = tmp_tensor.reshape(n, di, hi, wi, c1 * c0)
    return new_tensor


def shape_5d_2_6d(tensor):
    import math

    c0 = 16
    n, di, hi, wi, c = tensor.shape
    c1 = math.ceil(c / c0)
    tmp_tensor = tensor.reshape(n, di, hi, wi, c1, c0)
    new_tensor = np.transpose(tmp_tensor, axes=(0, 1, 4, 2, 3, 5))
    return new_tensor


def trans_nhwc_to_nc1hwc0(input_data, c0=16):
    in_shape = np.shape(input_data)
    axis_n = in_shape[0]
    axis_h = in_shape[1]
    axis_w = in_shape[2]
    axis_c = in_shape[3]
    axis_c0 = c0
    axis_c1 = _ceil_div(axis_c, axis_c0)
    c_pad = 0
    tmp_input_tensor = np.pad(
        input_data,
        ((0, 0), (0, 0), (0, 0), (0, c_pad)),
        mode="constant",
        constant_values=(0, 0),
    )
    tmp_input_tensor = tmp_input_tensor.reshape(
        axis_n, axis_h, axis_w, axis_c1, axis_c0
    )
    output_arr = np.transpose(tmp_input_tensor, axes=(0, 3, 1, 2, 4))
    return output_arr


def trans_nc1hwc0_to_nhwc(input_data):
    in_shape = np.shape(input_data)
    axis_n = in_shape[0]
    axis_c1 = in_shape[1]
    axis_h = in_shape[2]
    axis_w = in_shape[3]
    axis_c0 = in_shape[4]
    tmp_input_tensor = np.transpose(input_data, axes=(0, 2, 3, 1, 4))
    tmp_input_tensor = tmp_input_tensor.reshape(
        axis_n, axis_h, axis_w, axis_c1 * axis_c0
    )
    return tmp_input_tensor


def _high_dim_batch_to_space_nd(input_tensor, block_shape, crops):
    import tensorflow as tf

    M = block_shape.shape[0]
    input_shape = input_tensor.shape
    batch_size = input_shape[0]
    spatial_shape = input_shape[1 : 1 + M]
    remain_shape = input_shape[1 + M :]

    block_prod = tf.reduce_prod(block_shape)
    new_batch = batch_size // block_prod

    new_shape = tf.concat(
        [block_shape, [new_batch], spatial_shape, remain_shape], axis=0
    )
    reshaped = tf.reshape(input_tensor, new_shape)

    perm = [M]
    for i in range(M):
        perm.append(M + 1 + i)
        perm.append(i)
    remain_start = M + 1 + M
    perm.extend(range(remain_start, reshaped.shape.rank))
    transposed = tf.transpose(reshaped, perm)

    expanded_spatial = spatial_shape * block_shape
    new_shape2 = tf.concat([[new_batch], expanded_spatial, remain_shape], axis=0)
    merged = tf.reshape(transposed, new_shape2)

    begin = [0]
    size = [new_batch]
    for i in range(M):
        begin.append(crops[i, 0])
        size.append(expanded_spatial[i] - crops[i, 0] - crops[i, 1])
    R = input_tensor.shape.rank - 1 - M
    for i in range(R):
        begin.append(0)
        size.append(remain_shape[i])

    output = tf.slice(merged, begin, size)
    return output


def batch_to_space_nd_golden(x, block_shape, crops, **kwargs):
    """
    Kernel golden for batch_to_space_nd.
    All the parameters follow @batch_to_space_nd_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    """
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()

    data_x = x
    block_shape_arr = np.array(block_shape).astype(np.int64)
    crops_arr = np.array(crops).astype(np.int64)

    input_formats = kwargs.get("input_formats", ())
    fmt = input_formats[0] if input_formats and len(input_formats) > 0 else "ND"
    if isinstance(fmt, (list, tuple)):
        fmt = fmt[0] if fmt else "ND"

    if fmt == "NC1HWC0":
        data_x = trans_nc1hwc0_to_nhwc(data_x)
    elif fmt == "NDC1HWC0":
        data_x = shape_6d_2_5d(data_x)

    tensor_x = tf.placeholder(data_x.dtype, shape=data_x.shape)
    tf_block_shape = tf.constant(block_shape_arr)
    tf_crops = tf.constant(crops_arr)
    if block_shape_arr.shape[0] > 4:
        out = _high_dim_batch_to_space_nd(tensor_x, tf_block_shape, tf_crops)
    else:
        out = tf.batch_to_space_nd(tensor_x, tf_block_shape, tf_crops)

    with tf.Session() as sess:
        res = sess.run(out, feed_dict={tensor_x: data_x})

    if fmt == "NC1HWC0":
        res = trans_nhwc_to_nc1hwc0(res)
    elif fmt == "NDC1HWC0":
        res = shape_5d_2_6d(res)
    return res


class BatchToSpaceNdTfImpl:
    @staticmethod
    def _build(x_np, blocks_np, crops_np, input_format="ND"):
        import tensorflow as tf

        x = tf.convert_to_tensor(x_np)
        shape = tf.shape(x)
        if input_format == "NC1HWC0":
            if x.shape.rank != 5:
                raise ValueError("NC1HWC0 requires rank 5")
            x = tf.reshape(
                tf.transpose(x, [0, 2, 3, 1, 4]),
                [shape[0], shape[2], shape[3], shape[1] * shape[4]],
            )
        elif input_format == "NDC1HWC0":
            if x.shape.rank != 6:
                raise ValueError("NDC1HWC0 requires rank 6")
            x = tf.reshape(
                tf.transpose(x, [0, 1, 3, 4, 2, 5]),
                [shape[0], shape[1], shape[3], shape[4], shape[2] * shape[5]],
            )
        elif input_format != "ND":
            raise NotImplementedError(f"Unsupported input format: {input_format}")
        out = tf.raw_ops.BatchToSpaceND(
            input=x,
            block_shape=tf.cast(tf.convert_to_tensor(blocks_np), tf.int32),
            crops=tf.cast(tf.convert_to_tensor(crops_np), tf.int32),
        )
        out_shape = tf.shape(out)
        if input_format == "NC1HWC0":
            c0 = shape[4]
            out = tf.reshape(
                out, [out_shape[0], out_shape[1], out_shape[2], out_shape[3] // c0, c0]
            )
            out = tf.transpose(out, [0, 3, 1, 2, 4])
        elif input_format == "NDC1HWC0":
            c0 = shape[5]
            out = tf.reshape(
                out,
                [
                    out_shape[0],
                    out_shape[1],
                    out_shape[2],
                    out_shape[3],
                    out_shape[4] // c0,
                    c0,
                ],
            )
            out = tf.transpose(out, [0, 1, 4, 2, 3, 5])
        return out

    def __call__(self, x, block_shape, crops, **kwargs):
        import numpy as np
        import tensorflow as tf
        import torch

        def to_np(v):
            if not isinstance(v, torch.Tensor):
                return np.asarray(v)
            v = v.detach().cpu()
            if v.dtype == torch.bfloat16:
                import ml_dtypes

                return v.view(torch.int16).numpy().view(ml_dtypes.bfloat16)
            return v.numpy()

        x_np = np.asarray(to_np(x))
        blocks_np = np.asarray(to_np(block_shape)).reshape(-1)
        crops_np = np.asarray(to_np(crops)).reshape(-1, 2)
        formats = kwargs.get("input_formats", ())
        input_format = formats[0] if formats else "ND"
        if isinstance(input_format, (list, tuple)):
            input_format = input_format[0] if input_format else "ND"
        if tf.executing_eagerly():
            return [self._build(x_np, blocks_np, crops_np, input_format).numpy()]
        graph = tf.Graph()
        with graph.as_default():
            out_g = self._build(x_np, blocks_np, crops_np, input_format)
        with tf.compat.v1.Session(graph=graph) as sess:  # TF1 图模式兜底
            return [sess.run(out_g)]


class BatchToSpaceNdTestSpec:
    def golden(x, block_shape, crops, **kwargs):
        return [batch_to_space_nd_golden(x, block_shape, crops, **kwargs)]

    third_party = {"tf": BatchToSpaceNdTfImpl}

    tolerance = {
        "float32": {"standard": "binary_equal"},
        "float16": {"standard": "binary_equal"},
        "bfloat16": {"standard": "binary_equal"},
    }
