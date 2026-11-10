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


__golden__ = {"kernel": {"space_to_depth": "space_to_depth_golden"}}


__spec__ = {"space_to_depth": "SpaceToDepthTestSpec"}


def space_to_depth_golden(x, block_size, data_format="NHWC", **kwargs):
    """
    Kernel golden for space_to_depth.
    All the parameters follow @space_to_depth_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    """
    input_array0 = x
    shapes = input_array0.shape

    if data_format == "NCHW":
        n, c, h, w = shapes
        output_shapes = [n, c * (block_size**2), h // block_size, w // block_size]
        input_shapes = [n, c, h // block_size, block_size, w // block_size, block_size]
        perm = [0, 1, 3, 5, 2, 4]
    else:
        n, h, w, c = shapes
        output_shapes = [n, h // block_size, w // block_size, c * (block_size**2)]
        input_shapes = [n, h // block_size, block_size, w // block_size, block_size, c]
        perm = [0, 1, 3, 2, 4, 5]
    tmp = input_array0.reshape(input_shapes)
    tmp = np.transpose(tmp, perm)
    return tmp.reshape(output_shapes)


class SpaceToDepthImpl:
    def __call__(self, x, block_size, data_format="NHWC"):
        return self._impl_fn(x, block_size, data_format)

    def _impl_fn(self, x, block_size, data_format="NHWC"):
        import torch

        x_t = x if isinstance(x, torch.Tensor) else torch.from_numpy(x)
        if data_format == "NHWC":
            x_t = x_t.permute(0, 3, 1, 2)
        y = torch.nn.functional.pixel_unshuffle(x_t, block_size)
        n, c2, h2, w2 = y.shape
        c, bs = c2 // (block_size**2), block_size
        if data_format == "NHWC":
            y = (
                y.view(n, c, bs, bs, h2, w2)
                .permute(0, 4, 5, 2, 3, 1)
                .reshape(n, h2, w2, c2)
            )
            return y.contiguous()
        return y.contiguous()


class SpaceToDepthTestSpec:
    def golden(x, block_size, data_format="NHWC", **kwargs):
        return [space_to_depth_golden(x, block_size, data_format=data_format, **kwargs)]

    third_party = {"torch": SpaceToDepthImpl}

    tolerance = {
        "float32": {"standard": "binary_equal"},
        "float16": {"standard": "binary_equal"},
        "bfloat16": {"standard": "binary_equal"},
    }
