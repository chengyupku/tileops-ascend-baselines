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


# Asset modules are loaded by file path; resolve the shared helper independently of cwd/sys.path.
from pathlib import Path
from runpy import run_path

_framework_pad = run_path(
    str(Path(__file__).resolve().parents[4] / "tests" / "golden_helpers" / "padding.py")
)["framework_pad"]


__golden__ = {"kernel": {"mirror_pad": "mirror_pad_golden"}}

__spec__ = {"mirror_pad": "MirrorPadTestSpec"}


def _numpy_bfloat16():
    try:
        from ml_dtypes import bfloat16
    except ModuleNotFoundError:
        try:
            import tensorflow

            bfloat16 = tensorflow.bfloat16.as_numpy_dtype
        except ModuleNotFoundError:
            raise RuntimeError(
                "ml-dtypes or tensorflow is needed to support bfloat16 dtype!!! "
                "Please install with `pip3 install ml-dtypes` or `pip3 install tensorflow`"
            )
    return bfloat16


def _numpy_to_torch_tensor(np_array):
    import torch

    if np_array is None:
        return None
    np_dtype = np_array.dtype.name
    if "bfloat16" in np_dtype:
        np_int16 = np_array.view(dtype=np.int16)
        t_int16 = torch.from_numpy(np_int16)
        return t_int16.view(torch.bfloat16)
    else:
        return torch.from_numpy(np_array)


def _torch_to_numpy_tensor(torch_tensor):
    import torch

    if torch_tensor is None:
        return None
    if not isinstance(torch_tensor, torch.Tensor):
        raise RuntimeError(f"Only support torch.Tensor. But got {type(torch_tensor)}")
    torch_dtype = torch_tensor.dtype
    if torch_dtype == torch.bfloat16:
        t_int16 = torch_tensor.view(torch.int16)
        np_int16 = t_int16.numpy()
        return np_int16.view(dtype=_numpy_bfloat16())
    else:
        return torch_tensor.numpy()


def _pad_and_slice(arr, pad_width, mode="reflect"):
    pad_width = tuple(pad_width)
    if len(pad_width) != arr.ndim:
        raise ValueError(
            f"pad_width length ({len(pad_width)}) must match array ndim ({arr.ndim})"
        )

    result = arr

    for i, (left, right) in enumerate(pad_width):
        if left == 0 and right == 0:
            continue

        if left > 0 or right > 0:
            pad_tuple = [(0, 0)] * arr.ndim
            lv = left if left > 0 else 0
            rv = right if right > 0 else 0
            pad_tuple[i] = (lv, rv)
            result = _framework_pad(result, pad_tuple, mode)

        if left < 0 or right < 0:
            crop_left = abs(left) if left < 0 else 0
            crop_right = abs(right) if right < 0 else 0
            slice_start = crop_left
            slice_end = -crop_right if crop_right > 0 else None
            slices = [slice(None)] * arr.ndim
            slices[i] = slice(slice_start, slice_end)
            result = result[tuple(slices)]

    return result


def mirror_pad_golden(x, paddings, mode="REFLECT", **kwargs):
    """
    Kernel golden for mirror_pad.
    All the parameters follow @mirror_pad_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    """
    import torch

    pad_shape = np.reshape(paddings, (len(x.shape), 2))
    if mode == "REFLECT" or mode == "SYMMETRIC":
        if mode == "SYMMETRIC":
            np_mode = "symmetric"
        else:
            np_mode = "reflect"

        orig_dtype = x.dtype
        if orig_dtype.name == "bfloat16":
            x = _numpy_to_torch_tensor(x).to(torch.float32).numpy()

        neg_pad = _numpy_to_torch_tensor(np.array(pad_shape))
        neg_mask = neg_pad < 0
        indices = torch.nonzero(neg_mask, as_tuple=False)

        if indices.size(0) == 0:
            golden = _framework_pad(x, pad_shape, np_mode)
        else:
            golden = _pad_and_slice(x, pad_shape, mode=np_mode)

        if orig_dtype.name == "bfloat16":
            golden = _torch_to_numpy_tensor(
                _numpy_to_torch_tensor(golden).to(torch.bfloat16)
            )
    else:
        raise ValueError(f"Unsupported pad mode: {mode}")

    return golden


class MirrorPadImpl:
    def __call__(self, x, paddings, mode="REFLECT"):
        import torch

        mode = mode.lower()
        if mode != "reflect":
            raise NotImplementedError("PyTorch mirror pad only supports REFLECT mode")
        x_t = (
            x if isinstance(x, torch.Tensor) else _numpy_to_torch_tensor(np.asarray(x))
        )
        pads = torch.as_tensor(paddings).reshape(-1).tolist()
        pads = [[int(pads[2 * i]), int(pads[2 * i + 1])] for i in range(len(pads) // 2)]
        ndim = x_t.dim()
        offset = ndim - len(pads)

        # PyTorch 非 constant pad 仅支持指定 rank 的尾部 1～3 维。
        pad_width = [(0, 0)] * offset + pads
        active = [i for i, pair in enumerate(pad_width) if any(pair)]
        if not active:
            return x_t.clone()
        count = ndim - active[0]
        if count not in (1, 2, 3) or ndim not in (count + 1, count + 2):
            raise NotImplementedError(
                "PyTorch non-constant pad requires trailing 1-3 dimensions "
                "with one or two leading dimensions"
            )
        result = x_t
        for axis, (left, right) in enumerate(pad_width):
            if left > 0 or right > 0:
                flat = [0] * (2 * count)
                pos = 2 * (ndim - 1 - axis)
                flat[pos : pos + 2] = [max(0, left), max(0, right)]
                result = torch.nn.functional.pad(result, flat, mode="reflect")
            if left < 0 or right < 0:
                slices = [slice(None)] * ndim
                slices[axis] = slice(max(0, -left), right if right < 0 else None)
                result = result[tuple(slices)]
        return result


class MirrorPadTFImpl:
    def __call__(self, x, paddings, mode="REFLECT"):
        import tensorflow as tf
        import torch

        mode = mode.lower()
        if mode not in ("reflect", "symmetric"):
            raise NotImplementedError(f"TensorFlow pad does not support mode: {mode}")
        x_np = (
            _torch_to_numpy_tensor(x.detach().cpu())
            if isinstance(x, torch.Tensor)
            else np.asarray(x)
        )
        pads_np = (
            paddings.detach().cpu().numpy()
            if isinstance(paddings, torch.Tensor)
            else np.asarray(paddings)
        )
        pads = pads_np.astype(np.int64)
        pads = pads.reshape(-1, 2)
        if len(pads) > x_np.ndim:
            raise ValueError("paddings length exceeds input rank")
        pads = [(0, 0)] * (x_np.ndim - len(pads)) + pads.tolist()

        def build():
            result = tf.convert_to_tensor(x_np)
            for axis, (left, right) in enumerate(pads):
                slices = [slice(None)] * x_np.ndim
                slices[axis] = slice(max(0, -left), right if right < 0 else None)
                if left > 0 or right > 0:
                    widths = [[0, 0] for _ in range(x_np.ndim)]
                    widths[axis] = [max(0, left), max(0, right)]
                    result = tf.pad(
                        result,
                        widths,
                        mode=mode.upper(),
                    )
                result = result[tuple(slices)]
            return result

        if tf.executing_eagerly():
            out_np = build().numpy()
        else:
            graph = tf.Graph()
            with graph.as_default():
                out_g = build()
            with tf.compat.v1.Session(graph=graph) as sess:
                out_np = sess.run(out_g)
        return np.asarray(out_np)


class MirrorPadTestSpec:
    def golden(x, paddings, mode="REFLECT", **kwargs):
        return [mirror_pad_golden(x, paddings, mode, **kwargs)]

    third_party = {"torch": MirrorPadImpl, "tf": MirrorPadTFImpl}

    tolerance = {
        "float32": {"standard": "binary_equal"},
        "float16": {"standard": "binary_equal"},
        "bfloat16": {"standard": "binary_equal"},
    }
