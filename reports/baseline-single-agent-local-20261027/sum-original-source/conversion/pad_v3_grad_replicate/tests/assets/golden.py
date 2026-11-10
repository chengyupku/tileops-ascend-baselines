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

__golden__ = {"kernel": {"pad_v3_grad_replicate": "pad_v3_grad_replicate_golden"}}
__spec__ = {"pad_v3_grad_replicate": "PadV3GradReplicateTestSpec"}


def _parse_paddings(paddings, ndim, paddings_contiguous):
    if hasattr(paddings, "detach"):
        paddings = paddings.detach().cpu().numpy()
    pads = np.asarray(paddings, dtype=np.int64)
    pads = pads.reshape(-1, 2) if paddings_contiguous else pads.reshape(2, -1).T
    if len(pads) > ndim:
        raise ValueError("paddings length exceeds input rank")
    return [(0, 0)] * (ndim - len(pads)) + pads.tolist()


def pad_v3_grad_replicate_golden(
    x, paddings, mode="edge", paddings_contiguous=True, **kwargs
):
    if mode != "edge":
        raise NotImplementedError(f"Replication padding does not support mode: {mode}")
    pads = _parse_paddings(paddings, x.ndim, paddings_contiguous)
    out_shape = [size - left - right for size, (left, right) in zip(x.shape, pads)]
    if any(size <= 0 for size in out_shape):
        raise ValueError("Unpadded dimensions must be positive")
    result = x.astype(np.float32)
    # 逐维将输出梯度累加到 clamp(index-left, 0, input_size-1) 的输入位置。
    for axis, ((left, _), size) in enumerate(zip(pads, out_shape)):
        indices = np.clip(np.arange(result.shape[axis]) - left, 0, size - 1)
        moved = np.moveaxis(result, axis, 0)
        reduced = np.zeros((size,) + moved.shape[1:], dtype=np.float32)
        np.add.at(reduced, indices, moved)
        result = np.moveaxis(reduced, 0, axis)
    return result.astype(x.dtype)


class PadV3GradReplicateImpl:
    def __call__(self, x, paddings, mode="edge", paddings_contiguous=True, **kwargs):
        import torch

        if mode != "edge":
            raise NotImplementedError(
                f"PyTorch replication padding does not support mode: {mode}"
            )
        if not isinstance(x, torch.Tensor):
            x_np = np.asarray(x)
            x = (
                torch.from_numpy(x_np.view(np.int16)).view(torch.bfloat16)
                if x_np.dtype.name == "bfloat16"
                else torch.from_numpy(x_np)
            )
        pads = _parse_paddings(paddings, x.dim(), paddings_contiguous)
        active = [axis for axis, pair in enumerate(pads) if any(pair)]
        if not active:
            return x.clone()
        count = x.dim() - active[0]
        if count not in (1, 2, 3) or x.dim() not in (count + 1, count + 2):
            raise NotImplementedError(
                "PyTorch replication padding backward requires trailing 1-3 dimensions with one or two leading dimensions"
            )
        shape = [size - left - right for size, (left, right) in zip(x.shape, pads)]
        pad_list = [int(value) for pair in reversed(pads[-count:]) for value in pair]
        fn = getattr(torch.ops.aten, f"replication_pad{count}d_backward")
        return fn(x, x.new_empty(shape), pad_list)


class PadV3GradReplicateTestSpec:
    def golden(x, paddings, mode="edge", paddings_contiguous=True, **kwargs):
        return [
            pad_v3_grad_replicate_golden(
                x, paddings, mode, paddings_contiguous, **kwargs
            )
        ]

    third_party = {"torch": PadV3GradReplicateImpl}
    tolerance = {
        "float32": {"standard": "mix_tolerance"},
        "float16": {"standard": "mix_tolerance"},
        "bfloat16": {"standard": "mix_tolerance"},
    }
