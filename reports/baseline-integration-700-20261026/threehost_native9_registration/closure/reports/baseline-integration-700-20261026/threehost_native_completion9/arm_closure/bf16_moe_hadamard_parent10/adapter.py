"""Thin CPU-lazy contract binding for audited MoE and Hadamard V5 calls."""
from __future__ import annotations
import math

class V5Binding:
    def __init__(self, native_product):
        if not callable(native_product):
            raise TypeError("explicit native_product callable required")
        self.native_product = native_product

    def _product(self, lhs, rhs, *, metadata=None):
        import importlib
        torch = importlib.import_module("torch")
        if lhs.dtype != torch.float32 or rhs.dtype != torch.float32 or lhs.device != rhs.device:
            raise ValueError("native product requires same-device FP32 operands")
        y = self.native_product(lhs, rhs, metadata=metadata)
        if y.dtype != torch.float32 or y.device != lhs.device or tuple(y.shape) != (lhs.shape[0], rhs.shape[1]):
            raise ValueError("native product returned wrong shape/device/dtype")
        return y

    @staticmethod
    def _shape(x): return tuple(x.shape)

    @staticmethod
    def _zeros(shape, like, dtype):
        # Use the input tensor's device; torch import is intentionally lazy.
        import importlib
        torch = importlib.import_module("torch")
        return torch.zeros(shape, dtype=dtype, device=like.device)

    def moe(self, a, b, metadata, *, out_dtype=None, masked=False):
        """Original MoE contract: b is [E,N,K], metadata is ends or counts."""
        ends = metadata.tolist()
        if any(type(v) is not int for v in ends):
            raise ValueError("metadata must contain integers")
        ashape, bshape = self._shape(a), self._shape(b)
        if len(bshape) != 3 or len(ends) != bshape[0]:
            raise ValueError("metadata must have one entry per expert")
        if a.shape[-1] != bshape[-1] or a.device != b.device:
            raise ValueError("expert K/device contract mismatch")
        target_dtype = out_dtype if out_dtype is not None else a.dtype
        if masked:
            if len(ashape) != 3 or ashape[0] != bshape[0]:
                raise ValueError("masked layout requires a[E,M,K], b[E,N,K]")
            out = self._zeros((ashape[0], ashape[1], bshape[1]), a, target_dtype)
            if any(count < 0 or count > ashape[1] for count in ends):
                raise ValueError("masked count out of range")
            for e, count in enumerate(ends):
                if count < 0 or count > ashape[1]: raise ValueError("masked count out of range")
                if count:
                    lhs = a[e, :count, :].float()
                    rhs = b[e].transpose(0, 1).contiguous().float()
                    y = self._product(lhs, rhs, metadata={"expert":e,"count":count,"layout":"masked"})
                    out[e, :count, :] = y.to(target_dtype)
            return out
        if len(ashape) != 2 or len(ends) == 0: raise ValueError("physical layout requires a[P,K]")
        prev = 0
        for end in ends:
            if end < prev or end > ashape[0]: raise ValueError("physical_psum ends must be monotonic and bounded")
            prev = end
        prev = 0
        if ends[-1] != ashape[0]: raise ValueError("physical_psum final end must equal P")
        out = self._zeros((ashape[0], bshape[1]), a, target_dtype)
        for e, end in enumerate(ends):
            if end == prev: continue
            lhs = a[prev:end, :].float()
            rhs = b[e].transpose(0, 1).contiguous().float()
            y = self._product(lhs, rhs, metadata={"expert":e,"start":prev,"end":end,"layout":"physical_psum"})
            out[prev:end, :] = y.to(target_dtype)
            prev = end
        return out

    @staticmethod
    def _signs(n):
        return [[-1.0 if ((i & j).bit_count() & 1) else 1.0 for j in range(n)] for i in range(n)]

    @staticmethod
    def _coeff(rows, like):
        import importlib
        torch = importlib.import_module("torch")
        return torch.tensor(rows, dtype=torch.float32, device=like.device)

    def hadamard(self, x, *, n=None, base_order=1):
        n = int(n or x.shape[-1])
        if base_order != 1 or n not in (128, 4096) or x.shape[-1] != n:
            raise ValueError("only audited base_order=1 n=128/4096")
        rows = 1
        for d in x.shape[:-1]: rows *= int(d)
        xf = x.float()
        if n == 128:
            h = self._coeff(self._signs(128), x)
            first = self._product(xf.reshape(rows, 128), h, metadata={"stage":1,"n":128,"base_order":1})
            # Preserve original second K=N=1 native call for order-1.
            one = self._coeff([[1.0 / math.sqrt(128.0)]], x)
            second = self._product(first.reshape(rows * 128, 1), one, metadata={"stage":2,"n":128,"base_order":1,"k":1,"n_out":1})
            return second.reshape(x.shape).to(x.dtype)
        h64 = self._coeff(self._signs(64), x)
        first = self._product(xf.reshape(rows * 64, 64), h64, metadata={"stage":1,"n":4096,"radix":64})
        packed = first.reshape(rows, 64, 64).transpose(1, 2).contiguous().reshape(rows * 64, 64)
        hbase = self._coeff([[v / 64.0 for v in r] for r in self._signs(64)], x)
        second = self._product(packed, hbase, metadata={"stage":2,"n":4096,"radix":64,"normalization":1/64})
        return second.reshape(rows, 64, 64).transpose(1, 2).contiguous().reshape(x.shape).to(x.dtype)

def make_binding(native_product): return V5Binding(native_product)
