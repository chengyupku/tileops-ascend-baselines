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
"""Rsqrt multi-pathway golden in the TestSpec format.

通路支持表：

| 通路   | 支持 | 依据 |
|--------|------|------|
| kernel | ✅   | op_kernel/arch35 有 arch35 实现 |
| geir   | ✅   | op_graph/rsqrt_proto.h 有 REG_OP(Rsqrt) |
| aclnn  | ✅   | op_api/aclnn_rsqrt.cpp 暴露 aclnnRsqrt / aclnnInplaceRsqrt 符号 |
| e2e    | ✅   | torch_npu 中 torch.rsqrt 绑定到 aclnnRsqrt |
"""

import numpy as np
import torch

__spec__ = {
    "rsqrt": "RsqrtKernelSpec",
    "aclnnRsqrt": "RsqrtAclnnSpec",
    "aclnnInplaceRsqrt": "RsqrtInplaceAclnnSpec",
}

__golden__ = {
    "aclnn": {
        "aclnnInplaceRsqrt": "aclnn_inplace_rsqrt_golden",
        "aclnnRsqrt": "aclnn_rsqrt_golden",
    },
    "kernel": {"rsqrt": "rsqrt_golden"},
    "e2e": {"aclnnRsqrt": "aclnn_rsqrt_golden"},
}

_KERNEL_TOLERANCE = {
    "float16": {"standard": "cross_check", "level": "L1"},
    "float32": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
}


def _output_dtype(kwargs, index, default):
    output_dtypes = kwargs.get("output_dtypes") or []
    if index >= len(output_dtypes):
        return default
    dtype = output_dtypes[index]
    if isinstance(dtype, (list, tuple)):
        dtype = dtype[0]
    return str(dtype)


def rsqrt_golden(x, **kwargs):
    """Kernel golden for rsqrt. All input Tensors are numpy.ndarray."""
    ori_dtype = kwargs.get("input_dtypes", ["float32"])[0]
    x_dtype = x.dtype

    if ori_dtype and "bfloat16" in str(ori_dtype).lower():
        x_tensor = torch.from_numpy(x.astype(np.float32))
        output = torch.rsqrt(x_tensor)
        return output.numpy().astype(x_dtype, copy=False)
    elif ori_dtype and "float16" in str(ori_dtype).lower():
        x_tensor = torch.from_numpy(x.astype(np.float32))
        output = torch.rsqrt(x_tensor)
        return output.numpy().astype(x_dtype, copy=False)
    else:
        x_tensor = torch.from_numpy(x)
        output = torch.rsqrt(x_tensor)
        return output.numpy()


def _rsqrt_torch(x):
    """Compute rsqrt with fp32 upcast for half/bf16, matching kernel behavior."""
    x_dtype = x.dtype
    if x_dtype == torch.float16 or x_dtype == torch.bfloat16:
        y = torch.rsqrt(x.to(torch.float32))
        return y.to(x_dtype)
    return torch.rsqrt(x)


def aclnn_rsqrt_golden(self, out=None, **kwargs):
    """Aclnn golden for aclnnRsqrt. All input Tensors are torch.Tensor."""
    return _rsqrt_torch(self)


def aclnn_inplace_rsqrt_golden(selfRef=None, **kwargs):
    """Aclnn golden for aclnnInplaceRsqrt. All input Tensors are torch.Tensor."""
    return _rsqrt_torch(selfRef)


class _RsqrtCompose:
    """Third-party reference executed on the remote GPU server."""

    def __call__(self, self_=None, *args, **kwargs):
        del args, kwargs
        return [_rsqrt_torch(self_)]


class RsqrtKernelSpec:
    """kernel + geir shared spec. The golden entry receives numpy arrays."""

    @staticmethod
    def golden(x, **kwargs):
        return [rsqrt_golden(x, **kwargs)]

    third_party = {"torch": _RsqrtCompose}
    tolerance = _KERNEL_TOLERANCE


class RsqrtAclnnSpec:
    """aclnnRsqrt spec. The golden entry receives torch tensors."""

    @staticmethod
    def golden(self, out=None, **kwargs):
        del out
        return [_rsqrt_torch(self)]

    third_party = {"torch": _RsqrtCompose}
    tolerance = _KERNEL_TOLERANCE


class RsqrtInplaceAclnnSpec:
    """aclnnInplaceRsqrt spec. The golden entry receives torch tensors."""

    @staticmethod
    def golden(selfRef=None, **kwargs):
        return [_rsqrt_torch(selfRef)]

    third_party = {"torch": _RsqrtCompose}
    tolerance = _KERNEL_TOLERANCE
