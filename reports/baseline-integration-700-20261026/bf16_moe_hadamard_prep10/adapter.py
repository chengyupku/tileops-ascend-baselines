"""CPU-lazy thin adapters for the audited MoE/Hadamard V5 contracts.

No device or backend module is imported here.  ``native_product`` is mandatory:
it is the already-reviewed complete aclnnGroupedMatmulV5 product factory/callable.
This module only performs lossless layout/packing/scatter and fixed coefficient
packing; all products are delegated to that callable.
"""
from __future__ import annotations

import math


class V5Binding:
    """Thin binding around an explicitly supplied complete V5 product callable.

    The callable must accept ``(lhs, rhs, *, out_dtype, metadata)`` and execute
    the complete native product.  It may be a factory object with ``product``;
    no implicit import or fallback is attempted.
    """

    def __init__(self, native_product):
        if native_product is None or not callable(native_product):
            raise TypeError("native_product must be an explicitly supplied callable")
        self.native_product = native_product

    def _product(self, lhs, rhs, *, out_dtype=None, metadata=None):
        return self.native_product(lhs, rhs, out_dtype=out_dtype, metadata=metadata)

    @staticmethod
    def _zeros(shape, like, dtype=None):
        # torch and numpy are intentionally imported only after the caller binds
        # a CPU tensor.  No backend/device package is touched.
        if hasattr(like, "zeros_like_shape"):
            return like.zeros_like_shape(shape, dtype)
        mod = type(like).__module__
        if mod.startswith("torch"):
            import torch
            return torch.zeros(shape, dtype=dtype or like.dtype, device="cpu")
        import numpy as np
        return np.zeros(shape, dtype=dtype or getattr(like, "dtype", None))

    @staticmethod
    def _stack(parts, like, axis=0):
        if type(like).__module__.startswith("torch"):
            import torch
            return torch.stack(parts, dim=axis)
        import numpy as np
        return np.stack(parts, axis=axis)

    @staticmethod
    def _cat(parts, like, axis=0):
        if type(like).__module__.startswith("torch"):
            import torch
            return torch.cat(parts, dim=axis)
        import numpy as np
        return np.concatenate(parts, axis=axis)

    @staticmethod
    def _shape(x):
        return tuple(x.shape)

    def moe(self, a, b, counts, *, out_dtype=None, masked=False):
        """Run expert products using physical per-expert counts.

        ``counts[e]`` is the number of defined rows for expert e, never a
        prefix sum.  Contiguous layout is ``a[P,K], b[E,K,N]`` and output is
        ``[P,N]``.  Masked layout is ``a[E,M,K], b[E,K,N]`` and output keeps
        the full ``[E,M,N]`` shape; undefined padding is left untouched.
        """
        counts = [int(v) for v in counts]
        ashape, bshape = self._shape(a), self._shape(b)
        if len(bshape) != 3 or len(counts) != bshape[0]:
            raise ValueError("counts must contain one physical count per expert")
        if masked:
            if len(ashape) != 3 or ashape[0] != bshape[0]:
                raise ValueError("masked layout requires a[E,M,K] and b[E,K,N]")
            out = self._zeros((ashape[0], ashape[1], bshape[2]), a, dtype=out_dtype)
            for e, count in enumerate(counts):
                if count < 0 or count > ashape[1]:
                    raise ValueError("masked expert count outside max_m")
                if count == 0:
                    continue
                lhs = a[e, :count, :]
                rhs = b[e, :, :]
                y = self._product(lhs, rhs, out_dtype=out_dtype,
                                  metadata={"expert": e, "count": count, "layout": "masked"})
                out[e, :count, :] = y
            return out
        if len(ashape) != 2 or sum(counts) != ashape[0]:
            raise ValueError("contiguous physical counts must sum to P")
        out = self._zeros((ashape[0], bshape[2]), a, dtype=out_dtype)
        cursor = 0
        for e, count in enumerate(counts):
            if count < 0:
                raise ValueError("negative physical count")
            if count:
                y = self._product(a[cursor:cursor + count, :], b[e, :, :],
                                  out_dtype=out_dtype,
                                  metadata={"expert": e, "count": count,
                                            "layout": "physical_psum"})
                out[cursor:cursor + count, :] = y
            cursor += count
        return out

    @staticmethod
    def _hadamard_signs(n):
        # Fixed Sylvester sign convention used by the audited reference.
        return [[-1.0 if (i & j).bit_count() & 1 else 1.0
                 for j in range(n)] for i in range(n)]

    @staticmethod
    def _matrix(rows, like):
        if hasattr(like, "matrix_like"):
            return like.matrix_like((len(rows), len(rows[0]) if rows else 0))
        mod = type(like).__module__
        if mod.startswith("torch"):
            import torch
            return torch.tensor(rows, dtype=torch.float32, device="cpu")
        import numpy as np
        return np.asarray(rows, dtype=np.float32)

    def hadamard(self, x, *, n=None, base_order=1, out_dtype=None):
        """Fixed-order V5 products for n=4096 and n=128 audited layouts."""
        shape = self._shape(x)
        n = int(n or shape[-1])
        if base_order != 1 or n not in (128, 4096) or shape[-1] != n:
            raise ValueError("only audited base_order=1 n=128/4096 contracts are supported")
        rows = 1
        for d in shape[:-1]:
            rows *= int(d)
        if n == 128:
            # Original order-1 radix convention, represented as one fixed V5
            # product; no Python butterfly or reduction is introduced.
            h = self._matrix([[v / math.sqrt(n) for v in r]
                              for r in self._hadamard_signs(n)], x)
            flat = x.reshape(rows, n)
            return self._product(flat, h, out_dtype=out_dtype,
                                 metadata={"stage": 1, "n": 128, "base_order": 1}).reshape(shape)
        # Existing 64x64 two-stage layout: first unnormalised radix matrix,
        # transpose/pack, then normalized base matrix with 1/64 coefficient.
        h64 = self._matrix(self._hadamard_signs(64), x)
        first = self._product(x.reshape(rows * 64, 64), h64,
                              out_dtype="float32",
                              metadata={"stage": 1, "n": 4096, "radix": 64})
        packed = first.reshape(rows, 64, 64).transpose(0, 2, 1).reshape(rows * 64, 64)
        hbase = self._matrix([[v / 64.0 for v in r]
                              for r in self._hadamard_signs(64)], x)
        second = self._product(packed, hbase, out_dtype=out_dtype,
                               metadata={"stage": 2, "n": 4096, "radix": 64,
                                         "normalization": 1 / 64})
        return second.reshape(rows, 64, 64).transpose(0, 2, 1).reshape(shape)


def make_binding(native_product):
    """Explicit constructor; intentionally no default/native fallback."""
    return V5Binding(native_product)
