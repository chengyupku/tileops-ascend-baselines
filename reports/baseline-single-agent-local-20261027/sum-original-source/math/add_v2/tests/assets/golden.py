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
"""Kernel/GEIR golden for AddV2 using the TTK TestSpec format."""

import numpy as np
import torch


__spec__ = {
    # Kernel and GEIR share the snake-case registration and the same TestSpec.
    "add_v2": "AddV2KernelSpec",
}

# Retain the repository-facing legacy entry while consumers migrate to TestSpec.
__golden__ = {
    "kernel": {"add_v2": "add_v2_golden"},
}


_TOLERANCE = {
    "float16": {"standard": "cross_check", "level": "L1"},
    "float32": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
    "int8": {"standard": "binary_equal"},
    "uint8": {"standard": "binary_equal"},
    "int16": {"standard": "binary_equal"},
    "int32": {"standard": "binary_equal"},
    "int64": {"standard": "binary_equal"},
    # TTK cross_check currently supports float16/bfloat16/float32 only.
    "complex64": {"standard": "stat_rel_err"},
}


def _numpy_dtype(dtype):
    """Resolve TTK dtype values, including NumPy's optional bfloat16 dtype."""
    name = getattr(dtype, "name", str(dtype)).lower()
    if name in ("bf16", "bfloat16"):
        try:
            from ml_dtypes import bfloat16
        except ImportError as exc:
            raise RuntimeError(
                "AddV2 bfloat16 golden requires the optional ml-dtypes package"
            ) from exc
        return bfloat16
    return np.dtype(dtype)


def _output_dtypes(kwargs):
    values = kwargs.get("output_dtypes") or ()
    return [
        value[0] if isinstance(value, (list, tuple)) and value else value
        for value in values
    ]


def _to_torch(array):
    """Convert a contiguous NumPy tensor to torch without lowering precision."""
    array = np.asarray(array)
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    if array.dtype.name == "bfloat16":
        return torch.from_numpy(array.view(np.int16)).view(torch.bfloat16)
    return torch.from_numpy(array)


def _to_numpy(tensor):
    tensor = tensor.detach().cpu().contiguous()
    if tensor.dtype == torch.bfloat16:
        return tensor.view(torch.int16).numpy().view(_numpy_dtype("bfloat16"))
    return tensor.numpy()


def _compute(x1, x2, golden_mode=None):
    """Compute AddV2 through the PyTorch reference interface."""
    if x1.dtype != x2.dtype:
        raise ValueError(
            f"add_v2 only supports identical input dtypes, got x1={x1.dtype}, "
            f"x2={x2.dtype}"
        )
    output_dtype = x1.dtype
    if golden_mode != "Promote" and output_dtype in (torch.float16, torch.bfloat16):
        x1, x2 = x1.to(torch.float32), x2.to(torch.float32)
    output = torch.add(x1, x2)
    return [output if golden_mode == "Promote" else output.to(output_dtype)]


def _kernel_golden(x1, x2, **kwargs):
    if x1.dtype.name != x2.dtype.name:
        raise ValueError(
            f"add_v2 only supports identical input dtypes, got x1={x1.dtype}, "
            f"x2={x2.dtype}"
        )

    outputs = [
        _to_numpy(output)
        for output in _compute(_to_torch(x1), _to_torch(x2), kwargs.get("golden_mode"))
    ]
    if kwargs.get("golden_mode") == "Promote":
        return outputs
    output_dtypes = _output_dtypes(kwargs)
    return [
        output.astype(_numpy_dtype(output_dtypes[index]), copy=False)
        if index < len(output_dtypes)
        else output
        for index, output in enumerate(outputs)
    ]


class _AddV2Compose:
    """Torch GPU reference with the kernel's FP32 intermediates for FP16/BF16."""

    def __call__(self, x1, x2, **kwargs):
        del kwargs
        import torch

        output_dtype = x1.dtype
        if output_dtype in (torch.float16, torch.bfloat16):
            x1, x2 = x1.to(torch.float32), x2.to(torch.float32)
        return [torch.add(x1, x2).to(output_dtype)]


class AddV2KernelSpec:
    """TestSpec shared by the AddV2 kernel and GEIR pathways."""

    @staticmethod
    def golden(x1, x2, **kwargs):
        return _kernel_golden(x1, x2, **kwargs)

    third_party = {"torch": _AddV2Compose}
    tolerance = _TOLERANCE


def add_v2_golden(x1, x2, **kwargs):
    """Legacy kernel entry backed by the same TestSpec computation."""
    if not kwargs.get("output_dtypes"):
        kwargs = {**kwargs, "output_dtypes": (x1.dtype,)}
    return _kernel_golden(x1, x2, **kwargs)[0]


# 【不存在】ACLNN 通路：CMakeLists.txt 显式配置 ACLNNTYPE aclnn_exclude。
# 【不存在】e2e 通路：本算子不交付 ACLNN 符号，torch_npu 无对应绑定入口。
