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
"""Sign multi-pathway golden in the TestSpec format.

通路支持表：

| 通路   | 支持 | 依据 |
|--------|------|------|
| kernel | ✅   | op_kernel/arch35 有 arch35 实现 |
| geir   | ✅   | op_graph/sign_proto.h 有 REG_OP(Sign) |
| aclnn  | ✅   | op_api/aclnn_sign.cpp 暴露 aclnnSign 符号 |
| e2e    | ✅   | torch_npu 中 torch.sign 绑定到 aclnnSign |
"""

import numpy as np
import torch

__spec__ = {
    "sign": "SignKernelSpec",
    "aclnnSign": "SignAclnnSpec",
}

__golden__ = {
    "aclnn": {
        "aclnnSign": "aclnn_sign_golden",
    },
    "kernel": {"sign": "sign_golden"},
    "e2e": {"aclnnSign": "aclnn_sign_golden"},
}

_KERNEL_TOLERANCE = {
    "float16": {"standard": "cross_check", "level": "L1"},
    "float32": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
    "int8": {"standard": "binary_equal"},
    "int16": {"standard": "binary_equal"},
    "int32": {"standard": "binary_equal"},
    "int64": {"standard": "binary_equal"},
}


def _output_dtype(kwargs, index, default):
    output_dtypes = kwargs.get("output_dtypes") or []
    if index >= len(output_dtypes):
        return default
    dtype = output_dtypes[index]
    if isinstance(dtype, (list, tuple)):
        dtype = dtype[0]
    return str(dtype)


def sign_golden(x, **kwargs):
    """Kernel golden for sign. All input Tensors are numpy.ndarray."""
    ori_dtype = kwargs.get("input_dtypes", ["float32"])[0]
    x_dtype = x.dtype

    if ori_dtype and (
        "bfloat16" in str(ori_dtype).lower() or "float16" in str(ori_dtype).lower()
    ):
        x_tensor = torch.from_numpy(x.astype(np.float32))
        output = torch.sign(x_tensor)
        return output.numpy().astype(x_dtype, copy=False)
    else:
        x_tensor = torch.from_numpy(x)
        output = torch.sign(x_tensor)
        return output.numpy()


def aclnn_sign_golden(self, result=None, **kwargs):
    """Aclnn golden for aclnnSign. All input Tensors are torch.Tensor."""
    return [torch.sign(self)]


class _SignCompose:
    """Third-party reference executed on the remote GPU server."""

    def __call__(self, self_=None, *args, **kwargs):
        del args, kwargs
        return [torch.sign(self_)]


class SignKernelSpec:
    """kernel + geir shared spec. The golden entry receives numpy arrays."""

    @staticmethod
    def golden(x, **kwargs):
        return [sign_golden(x, **kwargs)]

    third_party = {"torch": _SignCompose}
    tolerance = _KERNEL_TOLERANCE


class SignAclnnSpec:
    """aclnnSign spec. The golden entry receives torch tensors."""

    @staticmethod
    def golden(self, result=None, **kwargs):
        del result
        return [torch.sign(self)]

    third_party = {"torch": _SignCompose}
    tolerance = _KERNEL_TOLERANCE
