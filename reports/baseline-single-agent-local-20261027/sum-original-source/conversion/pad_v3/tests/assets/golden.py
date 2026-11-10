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
from collections import deque


# Asset modules are loaded by file path; resolve the shared helper independently of cwd/sys.path.
from pathlib import Path
from runpy import run_path

_framework_pad = run_path(
    str(Path(__file__).resolve().parents[4] / "tests" / "golden_helpers" / "padding.py")
)["framework_pad"]


__golden__ = {"kernel": {"pad_v3": "pad_v3_golden"}}

__spec__ = {"pad_v3": "PadV3TestSpec"}


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


def _pad_and_slice(arr, pad_width, mode="edge"):
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


def pad_v3_golden(
    x,
    paddings,
    constant_values=None,
    mode="constant",
    paddings_contiguous=True,
    **kwargs,
):
    """
    Kernel golden for pad_v3.
    All the parameters follow @pad_v3_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    """
    import torch

    input_formats = kwargs.get("input_formats", ())
    x_format = input_formats[0] if input_formats and len(input_formats) > 0 else "ND"

    paddings = np.array(paddings).astype(np.int64)
    if paddings.ndim == 1:
        if paddings_contiguous:
            paddings = paddings.reshape(-1, 2)
        else:
            paddings = paddings.reshape(2, -1)

    if mode == "constant":
        pad_shape = deque()
        if paddings_contiguous:
            for i in range(len(paddings)):
                pad_shape.append(paddings[len(paddings) - 1 - i][0])
                pad_shape.append(paddings[len(paddings) - 1 - i][1])
        else:
            for i in range(len(paddings[0])):
                pad_shape.append(paddings[0][len(paddings[0]) - 1 - i])
                pad_shape.append(paddings[1][len(paddings[1]) - 1 - i])

        if x_format == "NC1HWC0":
            pad_shape.appendleft(0)
            pad_shape.appendleft(0)

        dtypes = {
            "uint8": "int8",
            "uint16": "int16",
            "uint32": "int32",
            "uint64": "int64",
        }

        if x.dtype.name in dtypes.keys():
            x_tensor = _numpy_to_torch_tensor(x.view(dtypes[x.dtype.name]))
        elif x.dtype.name == "bfloat16":
            x_tensor = _numpy_to_torch_tensor(x).to(torch.float32)
        else:
            x_tensor = _numpy_to_torch_tensor(x)

        if constant_values is not None:
            const_val = (
                constant_values.item()
                if isinstance(constant_values, np.ndarray)
                else constant_values
            )
        else:
            const_val = 0
        golden = torch.constant_pad_nd(x_tensor, tuple(pad_shape), const_val)

        if x.dtype.name in dtypes.keys():
            golden = golden.numpy().view(x.dtype)
        elif x.dtype.name == "bfloat16":
            golden = _torch_to_numpy_tensor(golden.to(torch.bfloat16))
        else:
            golden = golden.numpy()
    elif mode == "reflect" or mode == "symmetric" or mode == "edge":
        pad_shape = list()
        if paddings_contiguous:
            pad_shape = paddings
        else:
            pad_shape = np.stack(paddings, axis=1).ravel().tolist()
            pad_shape = np.array(pad_shape).reshape(-1, 2).tolist()

        orig_dtype = x.dtype
        if orig_dtype.name == "bfloat16":
            x = _numpy_to_torch_tensor(x).to(torch.float32).numpy()

        neg_pad = _numpy_to_torch_tensor(np.array(pad_shape))
        neg_mask = neg_pad < 0
        indices = torch.nonzero(neg_mask, as_tuple=False)

        if indices.size(0) == 0:
            golden = _framework_pad(x, pad_shape, mode)
        else:
            golden = _pad_and_slice(x, pad_shape, mode=mode)

        if orig_dtype.name == "bfloat16":
            golden = _torch_to_numpy_tensor(
                _numpy_to_torch_tensor(golden).to(torch.bfloat16)
            )
    else:
        raise ValueError(f"Unsupported pad mode: {mode}")

    return golden


class PadV3Impl:
    def __call__(
        self,
        x,
        paddings,
        constant_values=None,
        mode="constant",
        paddings_contiguous=True,
    ):
        import torch

        x_t = x if isinstance(x, torch.Tensor) else torch.from_numpy(x)
        pads = torch.as_tensor(paddings).reshape(-1).tolist()
        pads = [[int(pads[2 * i]), int(pads[2 * i + 1])] for i in range(len(pads) // 2)]
        if not paddings_contiguous:
            arr = torch.as_tensor(paddings).reshape(2, -1).tolist()
            pads = [[arr[0][i], arr[1][i]] for i in range(len(arr[0]))]
        ndim = x_t.dim()
        offset = ndim - len(pads)

        const_val = 0
        if constant_values is not None:
            cv = torch.as_tensor(constant_values).reshape(-1)
            const_val = cv[0].item() if cv.numel() else 0

        if mode == "symmetric":
            raise NotImplementedError("PyTorch pad does not support symmetric mode")
        if mode not in ("constant", "reflect", "edge"):
            raise NotImplementedError(f"PyTorch pad does not support mode: {mode}")

        if mode == "constant":
            pad_list = [v for pair in reversed(pads) for v in pair]
            return torch.constant_pad_nd(x_t, pad_list, const_val)

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
        torch_mode = "replicate" if mode == "edge" else "reflect"
        result = x_t
        for axis, (left, right) in enumerate(pad_width):
            if left > 0 or right > 0:
                flat = [0] * (2 * count)
                pos = 2 * (ndim - 1 - axis)
                flat[pos : pos + 2] = [max(0, left), max(0, right)]
                result = torch.nn.functional.pad(result, flat, mode=torch_mode)
            if left < 0 or right < 0:
                slices = [slice(None)] * ndim
                slices[axis] = slice(max(0, -left), right if right < 0 else None)
                result = result[tuple(slices)]
        return result


class PadV3TFImpl:
    def __call__(
        self,
        x,
        paddings,
        constant_values=None,
        mode="constant",
        paddings_contiguous=True,
    ):
        import tensorflow as tf
        import torch

        if mode not in ("constant", "reflect", "symmetric"):
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
        pads = pads.reshape(-1, 2) if paddings_contiguous else pads.reshape(2, -1).T
        if len(pads) > x_np.ndim:
            raise ValueError("paddings length exceeds input rank")
        pads = [(0, 0)] * (x_np.ndim - len(pads)) + pads.tolist()
        const_val = 0
        if constant_values is not None:
            if isinstance(constant_values, torch.Tensor):
                const_val = constant_values.item()
            else:
                const_val = np.asarray(constant_values).item()

        def build():
            result = tf.convert_to_tensor(x_np)
            for axis, (left, right) in enumerate(pads):
                slices = [slice(None)] * x_np.ndim
                slices[axis] = slice(max(0, -left), right if right < 0 else None)
                if mode == "constant":
                    result = result[tuple(slices)]
                if left > 0 or right > 0:
                    widths = [[0, 0] for _ in range(x_np.ndim)]
                    widths[axis] = [max(0, left), max(0, right)]
                    result = tf.pad(
                        result,
                        widths,
                        mode=mode.upper(),
                        constant_values=tf.convert_to_tensor(
                            const_val, dtype=result.dtype
                        ),
                    )
                if mode != "constant":
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


class PadV3TestSpec:
    def golden(
        x,
        paddings,
        constant_values=None,
        mode="constant",
        paddings_contiguous=True,
        **kwargs,
    ):
        return [
            pad_v3_golden(
                x, paddings, constant_values, mode, paddings_contiguous, **kwargs
            )
        ]

    third_party = {"torch": PadV3Impl, "tf": PadV3TFImpl}

    tolerance = {
        "float32": {"standard": "binary_equal"},
        "float16": {"standard": "binary_equal"},
        "bfloat16": {"standard": "binary_equal"},
    }
