"""CPU-lazy preparation only. Inject an audited complete V5 dense_fp32 factory.

No registration or default provider. Path loading requires a parent-approved hash;
this pins Python bytes only, NOT native dependencies. Parent must attest those.
The injected factory owns full TensorList API, widening, output cast and cleanup.
"""
from pathlib import Path
from types import SimpleNamespace, ModuleType
import hashlib


def load_factory(path, *, sha256, symbol='dense_fp32'):
    path = Path(path).resolve(strict=True)
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if not sha256 or digest != sha256:
        raise ValueError('Native factory source pin mismatch')
    # No fixed sys.modules alias, no sys.path mutation, no fallback.
    module = ModuleType('_reviewed_v5_' + digest)
    module.__file__ = str(path)
    exec(compile(data, str(path), 'exec'), module.__dict__)
    factory = getattr(module, symbol)
    if not callable(factory):
        raise TypeError('Factory must be callable')
    return factory


def make_runner(operator, params, *, native_factory, tensor_api=None):
    """Return (run, native_candidate); factory is explicit and never inferred.

    params must be the exact manifest parameters, including T and bias presence.
    NN supported for contract tests; assigned Grouped keys are NT only. TN/TT stop.
    """
    p = dict(params)
    if p.get('T') != 'bfloat16':
        raise ValueError('This preparation is explicitly BF16 only')
    if operator not in ('BmmFwdOp', 'Conv1dFwdOp', 'GroupedGemmFwdOp'):
        raise NotImplementedError(operator)
    if operator == 'GroupedGemmFwdOp' and p['transpose_a']:
        raise NotImplementedError('TN/TT outside reviewed thin-binding scope')
    if operator == 'Conv1dFwdOp' and (p['groups'] != 1 or
            not all(isinstance(p[k], int) for k in ('stride', 'padding', 'dilation'))):
        raise NotImplementedError('Only reviewed integer ungrouped Conv1d packing')
    if not callable(native_factory):
        raise TypeError('Explicit reviewed native factory is required')
    if tensor_api is None:
        import torch as tensor_api
    t = tensor_api
    native_op = 'BmmFwdOp' if operator == 'BmmFwdOp' else 'GemmFwdOp'
    proxy = SimpleNamespace(_call=SimpleNamespace(signature=SimpleNamespace(name=native_op)),
                            params={'trans_a': False, 'trans_b': False})
    inner = native_factory(proxy)
    product = inner.implementation.run

    def check(x, shape, dtype=t.bfloat16):
        if tuple(x.shape) != tuple(shape) or x.dtype != dtype:
            raise ValueError(f'Tensor contract mismatch: {x.shape}, {x.dtype}')

    def result(y, shape):
        check(y, shape)
        return y

    def bmm(a, b):
        check(a, (p['B'], p['M'], p['K']))
        check(b, (p['B'], p['K'], p['N']))
        # One call passes full batch to original Bmm TensorList branch.
        return result(product(a, b), (p['B'], p['M'], p['N']))

    def conv(x, w, bias=None):
        check(x, (p['N'], p['C_in'], p['L_in']))
        check(w, (p['C_out'], p['C_in'], p['kW']))
        if (bias is not None) != p['bias']:
            raise ValueError('Manifest bias presence mismatch')
        if bias is not None:
            check(bias, (p['C_out'],))
        st, pad, dil = p['stride'], p['padding'], p['dilation']
        x = t.nn.functional.pad(x, (pad, pad))
        length = (x.shape[2] - dil * (w.shape[2] - 1) - 1) // st + 1
        windows = x.as_strided((x.shape[0], x.shape[1], length, w.shape[2]),
                              x.stride()[:2] + (x.stride()[2] * st, x.stride()[2] * dil))
        lhs = windows.permute((0, 2, 1, 3)).contiguous().reshape(-1, w[0].numel())
        rhs = w.reshape(w.shape[0], -1).T.contiguous()
        # Bias belongs to native optional TensorList, never a Python add.
        y = product(lhs, rhs, bias)
        result(y, (p['N'] * length, p['C_out']))
        return y.reshape((x.shape[0], length, w.shape[0])).movedim(-1, 1).contiguous()

    def grouped(a, b, sizes, offsets):
        check(a, (p['M'], p['K']))
        check(b, (p['G'], p['N'], p['K']) if p['transpose_b'] else
                 (p['G'], p['K'], p['N']))
        check(sizes, (p['G'],), t.int32)
        check(offsets, (p['G'],), t.int32)
        ns, starts = sizes.tolist(), offsets.tolist()
        end = 0
        for n, start in zip(ns, starts, strict=True):
            if type(n) is not int or type(start) is not int or n < 0 or start != end:
                raise ValueError('sizes/offsets violate canonical prefix invariant')
            end += n
        if end != p['M']:
            raise ValueError('sizes do not cover complete output')
        out = t.empty((p['M'], p['N']), dtype=a.dtype, device=a.device)
        for i, (n, start) in enumerate(zip(ns, starts, strict=True)):
            if n == 0:
                continue  # No call, no invented output row, no shifted weight index.
            rhs = b[i].T if p['transpose_b'] else b[i]
            y = product(a[start:start+n].contiguous(), rhs.contiguous())
            out[start:start+n] = result(y, (n, p['N']))
        return out

    return {'BmmFwdOp': bmm, 'Conv1dFwdOp': conv, 'GroupedGemmFwdOp': grouped}[operator], inner


def candidate(case, *, native_factory):
    """Parent-only future adapter; no shared registry updates here."""
    from dataclasses import replace
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import source_hashes
    p = dict(case._call.indices)
    if case._call.signature.name == 'Conv1dFwdOp':
        p['bias'] = case._call.present('bias')
    run, inner = make_runner(case._call.signature.name, p, native_factory=native_factory)
    return replace(inner, implementation=Implementation(run),
                   source_sha256=inner.source_sha256 | source_hashes(Path(__file__)),
                   adaptation='Complete original V5 factory; batch TensorList, original window packing/native bias, canonical-prefix grouped slicing/scatter; no new input mathematics.')


def make_candidate(case):
    import json
    key=('ops-transformer',case._call.signature.name,case.id)
    task=Path(__file__).resolve().parent
    rows=json.loads((task/'CASES.json').read_text())
    if key not in [tuple(r['key']) for r in rows]:
        raise ValueError('Outside exact FP16 scope')
    path=task/'binding.py'
    pins=json.loads((task/'FROZEN.json').read_text())['sha256']
    return candidate(case,native_factory=load_factory(path,sha256=pins[str(path)]))
