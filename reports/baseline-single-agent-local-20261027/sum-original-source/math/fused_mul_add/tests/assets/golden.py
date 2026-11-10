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

__spec__ = {"fused_mul_add": "FusedMulAddKernelSpec"}

__golden__ = {"kernel": {"fused_mul_add": "fused_mul_add_golden"}}

_KERNEL_TOLERANCE = {
    "float16": {"standard": "cross_check", "level": "L1"},
    "float32": {"standard": "cross_check", "level": "L1"},
    "int32": {"standard": "binary_equal"},
}


def fused_mul_add_golden(x1, x2, x3, **kwargs):
    """
    Kernel golden for fused_mul_add.
    All the parameters follow @fused_mul_add_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    """
    # y = x1 * x2 + x3, with NumPy broadcasting along ND format.
    dtype = x1.dtype
    if dtype == np.int32:
        # Integer path: match the kernel's explicit Mul -> Add (no fp32 lift).
        out = (x1 * x2 + x3).astype(np.int32)
    else:
        # Float / half path: lift to fp32 for the intermediate compute, then cast back.
        out = x1.astype("float32") * x2.astype("float32") + x3.astype("float32")
        out = out.astype(dtype)
    return out


class _FusedMulAddCompose:
    """Third-party reference executed on the remote GPU server."""

    def __call__(self, x1, x2, x3, **kwargs):
        del kwargs
        import torch  # the remote server executes this, the local golden stays numpy-only

        return [torch.add(torch.mul(x1, x2), x3)]


class FusedMulAddKernelSpec:
    """kernel spec: numpy golden + third-party reference + precision standard."""

    @staticmethod
    def golden(x1, x2, x3, **kwargs):
        return [fused_mul_add_golden(x1, x2, x3, **kwargs)]

    third_party = {"torch": _FusedMulAddCompose}
    tolerance = _KERNEL_TOLERANCE
