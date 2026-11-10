"""Accepted ops-nn factories; module import never imports any device runtime."""
import json
from pathlib import Path


def _provider(name, accepted_ids, pinned_ids, continuation_ids):
    def factory(case):
        from benchmarks.ascend_baselines import UnsupportedCase
        if case._call.signature.name!=name or case.id not in accepted_ids:
            raise UnsupportedCase('This exact manifest case has no accepted ops-nn numerical and native-ownership receipt; use staging for validation.')
        # Original-only route boundary: historical composed receipts remain intact.
        if name in ('LogicalAndFwdOp', 'LogicalOrFwdOp', 'LogicalNotFwdOp', 'BitwiseNotFwdOp', 'StdFwdOp', 'AdaLayerNormFwdOp', 'AdaLayerNormZeroFwdOp', 'FusedAddRMSNormFwdOp', 'FusedAddLayerNormFwdOp'):
            raise RuntimeError('Retired composed ops-nn adapter route; historical receipts are preserved. This is not a general original-API absence determination.')
        if name in ('VarFwdOp', 'VarMeanFwdOp') and int(case.params.get('correction', 1)) != 0:
            raise RuntimeError('Retired variance correction composition; original zero-correction auxiliary-output route remains available.')
        if name in ('ClampScalarFwdOp', 'ClampTensorFwdOp', 'HardtanhFwdOp'):
            both_bounds = name == 'HardtanhFwdOp' or (case.params.get('min') is not None and case.params.get('max') is not None)
            if both_bounds:
                raise RuntimeError('Retired two-bound max/min composition; single-bound original API routes remain available.')
        if name in ('CausalConv1dFwdOp', 'CausalConv1dUpdateFwdOp') and case.params.get('activation') == 'silu':
            raise RuntimeError('Retired Conv2DV2 plus Swish composition; original non-SiLU route remains available.')
        if name == 'PagedKVCacheGatherFwdOp' and case.params.get('scale') is not None:
            raise RuntimeError('Retired GatherV2 plus scale multiplication composition; original unscaled route remains available.')
        # Frozen receipts remain evidence; affected MatMulV3 entry is retired.
        if name in {"GroupedGemmFwdOp", "GemmFP8FwdOp", "BmmFP8FwdOp",
                    "MoEGroupedGemmFwdOp", "GemmW4A16FwdOp", "MoEGroupedGemmFP8FwdOp"}:
            import hashlib
            import os
            from benchmarks.ascend_baselines import WORKSPACE
            vendor = Path(os.environ.get("OPS_NN_VENDOR", str(WORKSPACE / ".cache/all-pairs/ops_nn/install3/vendors/allpairs_nn")))
            for proto in vendor.glob("op_proto/lib/linux/*/libcust_opsproto_rt2.0.so"):
                if hashlib.sha256(proto.read_bytes()).hexdigest() == "a7e6ccf94e57413d20c0b4b529c5e10b815331f3dab8e9b4a828f6b06ec5bf7f":
                    raise RuntimeError("MatMulV3 deployment blocked: selected proto contains internal registration modification; deploy pinned original before reuse. This is not an unsupported-case determination.")
        if case.id in continuation_ids:
            import os
            from .continuation_runtime import accepted_records,WS
            record=accepted_records().get((name,case.id))
            if record is None:
                raise UnsupportedCase('Continuation receipt/runtime source hashes no longer match; revalidation is required.')
            if Path(os.environ.get('OPS_NN_VENDOR','')).resolve()!=(WS/record['required_vendor']).resolve():
                raise UnsupportedCase('This exact continuation case requires its recorded install3 vendor before runtime initialization.')
            from .staging import CANDIDATES
            result=CANDIDATES[name](case)
            if result.api!=record['api'] or any(record['source_sha256'].get(k)!=v for k,v in result.source_sha256.items()):
                raise UnsupportedCase('Current candidate differs from its accepted complete receipt.')
            return result
        if case.id in pinned_ids:
            import os
            from dataclasses import replace
            from .preserved_runtime import accepted_records,WS
            if (name,case.id) not in accepted_records():
                raise UnsupportedCase('Preserved runtime source/dependency/receipt hashes no longer match; revalidation is required.')
            required=WS/'.cache/all-pairs/ops_nn/install3/vendors/allpairs_nn'
            if Path(os.environ.get('OPS_NN_VENDOR','')).resolve()!=required:
                raise UnsupportedCase('Preserved smoke004 runtime requires OPS_NN_VENDOR=install3/vendors/allpairs_nn before runtime initialization; see runtime_env.json.')
            from .pinned_smoke004.staging import CANDIDATES as PINNED
            result=PINNED[name](case)
            return replace(result,adaptation='Byte-preserved smoke004 runtime, accepted for this exact case with full output/input/native evidence; relocation introduces no new numerical claim. '+result.adaptation)
        # Historical receipts bind their original factory and classic vendor.
        # A newly validated staging route must not inherit those old case passes.
        from .catalog import LEGACY
        if name in LEGACY:
            from .matrix import legacy
            return legacy(case)
        from .staging import CANDIDATES
        return CANDIDATES[name](case)
    factory.__name__='accepted_'+name
    return factory


_file=Path(__file__).with_name('accepted_cases.json')
_accepted=json.loads(_file.read_text()) if _file.exists() else {}
_pinned_file=Path(__file__).with_name('accepted_pinned_cases.json')
_pinned=json.loads(_pinned_file.read_text()) if _pinned_file.exists() else {}
_continuation_file=Path(__file__).with_name('accepted_continuation_cases.json')
_continuation_data=json.loads(_continuation_file.read_text()) if _continuation_file.exists() else {'records':[]}
_continuation={}
for _record in _continuation_data['records']:
    _continuation.setdefault(_record['op'],set()).add(_record['case_id'])
PROVIDERS={name:_provider(name,frozenset(ids),frozenset(_pinned.get(name,())),frozenset(_continuation.get(name,()))) for name,ids in _accepted.items()}

# Add reviewed exact keys without replacing historical acceptance or fallbacks.
from .verified_runtime import install as _install_verified
_install_verified(PROVIDERS)



# Exact numerical receipt: baseline-integration-700-20261026/nn_math/w4graph01
from .integration700_nn_math_w4graph01 import make as _nn_math_w4graph01
_nn_math_old_w4graph01 = PROVIDERS.get('GemmW4A16FwdOp')
def _nn_math_route_w4graph01(case, _prior=_nn_math_old_w4graph01):
    if case._call.signature.name == 'GemmW4A16FwdOp' and case.id == 'decode-l2-resident-float16':
        return _nn_math_w4graph01(case)
    if _prior is not None:
        return _prior(case)
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase('No accepted exact key')
PROVIDERS['GemmW4A16FwdOp'] = _nn_math_route_w4graph01
