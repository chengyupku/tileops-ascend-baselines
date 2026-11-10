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

__spec__ = {
    "mul_no_nan": "MulNoNanKernelSpec",
    "tf.raw_ops.MulNoNan": "MulNoNanTensorFlowSpec",
}

__golden__ = {"kernel": {"mul_no_nan": "mul_no_nan_golden"}}

_KERNEL_TOLERANCE = {
    "float16": {"standard": "cross_check", "level": "L1"},
    "float32": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
    "int32": {"standard": "binary_equal"},
}

_TENSORFLOW_TOLERANCE = {
    "float16": {"standard": "cross_check", "level": "L1"},
    "float32": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
}


def mul_no_nan_golden(x1, x2, **kwargs):
    """
    Kernel golden for mul_no_nan.
    All the parameters follow @mul_no_nan_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.

    Semantics: y = (x2 == 0) ? 0 : x1 * x2 (element-wise, NumPy broadcasting).
    The mask is on x2 ONLY, so wherever x2 == 0 the product is replaced by 0
    regardless of what that product would have been -- including NaN produced
    by x1 == inf (inf * 0) or by x1 == NaN. Conversely, where x2 != 0 the
    product is returned as is, so x1 == 0 with x2 == inf still yields NaN and
    NOT 0. This is the core differentiator vs plain Mul.
    """
    import tensorflow as tf

    with tf.device("/CPU:0"):
        x, y = (
            tf.convert_to_tensor(np.asarray(x1)),
            tf.convert_to_tensor(np.asarray(x2)),
        )
        output_dtype = x.dtype
        promote = kwargs.get("golden_mode") == "Promote"
        if not promote and output_dtype in (tf.float16, tf.bfloat16):
            x, y = tf.cast(x, tf.float32), tf.cast(y, tf.float32)
        if output_dtype.is_integer:
            # The native TF op excludes int32; its primitive composition has
            # the same mask-on-x2 and native integer overflow semantics.
            product = tf.multiply(x, y)
            output = tf.where(tf.equal(y, 0), tf.zeros_like(product), product)
        else:
            output = tf.raw_ops.MulNoNan(x=x, y=y)
        if not promote:
            output = tf.cast(output, output_dtype)
        return np.asarray(output.numpy())


class _MulNoNanCompose:
    """Third-party reference executed on the remote GPU server."""

    def __call__(self, x1, x2, **kwargs):
        del kwargs
        import torch

        output_dtype = x1.dtype
        if output_dtype in (torch.float16, torch.bfloat16):
            x1, x2 = x1.to(torch.float32), x2.to(torch.float32)
        prod = torch.mul(x1, x2)
        output = torch.where(torch.eq(x2, 0), torch.zeros_like(prod), prod)
        return [output.to(output_dtype)]


class MulNoNanKernelSpec:
    """Kernel/GEIR: TensorFlow CPU golden and a Torch GPU competitor."""

    @staticmethod
    def golden(x1, x2, **kwargs):
        return [mul_no_nan_golden(x1, x2, **kwargs)]

    third_party = {"torch": _MulNoNanCompose}
    tolerance = _KERNEL_TOLERANCE


class _MulNoNanE2ECompose:
    """Use the TF API's parameter names for TTK's name-based GPU dispatch."""

    def __call__(self, x, y, **kwargs):
        return _MulNoNanCompose()(x, y, **kwargs)


class MulNoNanTensorFlowSpec:
    """TensorFlow E2E spec registered by the CSV ``api_name``."""

    # Omit a custom golden: TTK's native CPU TF fallback applies Promote to
    # raw inputs.  Its custom E2E plugin path bypasses that promotion.
    third_party = {"torch": _MulNoNanE2ECompose}
    # tf.raw_ops.MulNoNan itself supports floating/complex tensors, not int32.
    # This CANN operator exposes only the three floating types from that overlap.
    tolerance = _TENSORFLOW_TOLERANCE
