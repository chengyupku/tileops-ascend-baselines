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
import torch

__golden__ = {
    "aclnn": {
        "aclnnInplaceUniform": "aclnn_inplace_uniform_golden",
    }
}


def aclnn_inplace_uniform_golden(
    selfRef, from_val=0, to_val=0, seed=0, offset=0, **kwargs
):
    """
    Aclnn golden for aclnnInplaceUniform.
    """
    import tensorflow as tf

    if hasattr(seed, "item"):
        seed = seed.item()
    if hasattr(offset, "item"):
        offset = offset.item()

    seed_list = [seed]
    offset_list = [0, offset]

    attrs = kwargs.get("attributes", {})
    start = attrs.get("from", from_val) if attrs else from_val
    end = attrs.get("to", to_val) if attrs else to_val

    dtype = str(selfRef.dtype)[6:]

    end_const = tf.cast(tf.constant(end), dtype)
    start_const = tf.cast(tf.constant(start), dtype)
    input_shape = list(selfRef.shape)

    uniform_data = tf.raw_ops.StatelessRandomUniformV2(
        shape=input_shape, key=seed_list, counter=offset_list, alg=1, dtype=dtype
    )
    mul_data = tf.multiply(uniform_data, (end_const - start_const))
    add_data = tf.add(mul_data, start_const)
    output_data = tf.cast(add_data, dtype).numpy()
    output_data = torch.from_numpy(output_data)

    return output_data
