"""SGL's own implementations; importing this registry never imports torch/NPU."""
from .elementwise import sigmoid, silu

PROVIDERS = {'SigmoidFwdOp': sigmoid, 'SiluFwdOp': silu}


# SGL linalg36: verified exact keys only; preserve every preexisting route.
from .integration700_sgl_linalg_register36 import build as _integration700_sgl_linalg_register36_build
_integration700_sgl_linalg_register36_keys = {'BmmFwdOp': ['mha-decode-pv-bfloat16', 'mha-decode-pv-float16', 'mha-decode-qk-bfloat16', 'mha-decode-qk-float16', 'moe-prefill-bfloat16', 'square-128-bfloat16', 'square-128-float16', 'square-1k-bfloat16', 'square-1k-float16', 'square-512-bfloat16', 'square-512-float16'], 'GemmFwdOp': ['ds-v3-decode-down-bfloat16', 'ds-v3-decode-qkv-a-bfloat16', 'ds-v3-m1-qkv-a-bfloat16', 'ds-v3-m16-mlp-up-bfloat16', 'ds-v3-m2-qkv-a-bfloat16', 'ds-v3-m32-mlp-up-bfloat16', 'ds-v3-m64-down-bfloat16', 'ds-v3-m96-qkv-a-bfloat16', 'ds-v3-prefill-down-bfloat16', 'ds-v3-prefill-mlp-up-bfloat16', 'ds-v3-prefill-mlp-up-float16', 'ds-v3-prefill-o-proj-bfloat16', 'ds-v3-prefill-q-b-bfloat16', 'ds-v3-prefill-qkv-a-bfloat16', 'qwen3-8b-down-prefill-bfloat16', 'qwen3-8b-gate-up-b32-bfloat16', 'qwen3-8b-gate-up-decode-bfloat16', 'square-1k-bfloat16', 'vit-l16-in21k-head-bfloat16'], 'GroupedGemmFwdOp': ['moe-dgrad-float16', 'moe-fwd-bfloat16', 'moe-fwd-float16', 'qwen3-30b-ep8-empty-bfloat16', 'qwen3-30b-ep8-ragged-bfloat16', 'qwen3-30b-ep8-routed1k-bfloat16']}
def _integration700_sgl_linalg_register36_route(previous, exact_keys):
    def route(case):
        if case.id in exact_keys:
            return _integration700_sgl_linalg_register36_build(case)
        if previous is not None:
            return previous(case)
        from benchmarks.ascend_baselines import UnsupportedCase
        raise UnsupportedCase('No SGL provider for this unregistered exact key')
    return route
for _integration700_sgl_linalg_register36_op, _integration700_sgl_linalg_register36_cases in _integration700_sgl_linalg_register36_keys.items():
    PROVIDERS[_integration700_sgl_linalg_register36_op] = _integration700_sgl_linalg_register36_route(
        PROVIDERS.get(_integration700_sgl_linalg_register36_op), frozenset(_integration700_sgl_linalg_register36_cases))
# Exact numerically verified SGL attention recovery keys; preserve all prior routes.
from pathlib import Path as _ar14_Path
_ar14_root = _ar14_Path('/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/sgl_attention_register14')
_ar14_keys = {'GQAPrefillPagedWithKVCacheFwdOp': ['qwen35-9b-rope-float16'], 'MHADecodePagedWithKVCacheFwdOp': ['serving-kv1k-float16', 'serving-kv512-float16'], 'MLAKVCacheWriteFwdOp': ['ds-v3-decode-bfloat16-bfloat16', 'kimi-k2-rope-bfloat16-bfloat16-bfloat16'], 'DSADecodeWithKVCacheFwdOp': ['mla-d512-h128-kv8k-bfloat16', 'mla-d576-h128-kv8k-bfloat16'], 'GQAVarlenFwdOp': ['gemma4-26b-a4b-ragged-bfloat16', 'gemma4-31b-uniform-float16'], 'MLADecodeWithKVCacheFwdOp': ['ds-v2-32k-float16', 'mla-b128-16384-bfloat16', 'mla-b128-16384-float16', 'mla-b128-32768-bfloat16', 'mla-b128-32768-float16']}
def _ar14_route(op, keys, previous):
    def route(case):
        if case.id not in keys:
            if previous is not None:
                return previous(case)
            from benchmarks.ascend_baselines import UnsupportedCase
            raise UnsupportedCase('No registered SGL attention implementation for this key')
        import sys, importlib
        import benchmarks.ascend_baselines as base
        saved = base.LIBRARIES
        base.LIBRARIES = _ar14_root / 'libraries'
        sys.path.insert(0, str(_ar14_root))
        try:
            module = importlib.import_module('integration700_sgl_attention_register14')
            assert _ar14_Path(module.__file__).resolve().parent == _ar14_root
            return module.PROVIDERS[op](case)
        finally:
            base.LIBRARIES = saved
            sys.path.remove(str(_ar14_root))
    return route
for _ar14_op, _ar14_cases in _ar14_keys.items():
    PROVIDERS[_ar14_op] = _ar14_route(_ar14_op, frozenset(_ar14_cases), PROVIDERS.get(_ar14_op))

# sgl_gdn_register30: exact verified original complete API bindings, preserve prior routes.
from .integration700_sgl_gdn_register30 import build as _integration700_sgl_gdn_register30_build
_integration700_sgl_gdn_register30_keys = {'GDNFwdOp': ['decode-b1-bfloat16', 'decode-b8-bfloat16', 'decode-b8-float16', 'decode-fresh-bfloat16', 'decode-vfirst-b17-bfloat16', 'q35-gva-decode-b1-bfloat16', 'q35-gva-decode-b1-float16', 'q35-gva-decode-b16-bfloat16', 'q35-gva-decode-b16-float16', 'q35-gva-decode-b256-bfloat16', 'q35-gva-decode-b256-float16', 'q35-gva-decode-b32-bfloat16', 'q35-gva-decode-b32-float16', 'q35-gva-decode-b4-bfloat16', 'q35-gva-decode-b4-float16', 'q35-gva-decode-b512-bfloat16', 'q35-gva-decode-b512-float16', 'q35-gva-decode-b8-bfloat16', 'q35-gva-decode-b8-float16', 'decode-b1-float16', 'q35-gva-decode-b128-bfloat16', 'q35-gva-decode-b128-float16', 'q35-gva-decode-b64-bfloat16', 'q35-gva-decode-b64-float16'], 'DeltaNetFwdOp': ['decode-continued-bfloat16', 'decode-continued-float16', 'decode-fresh-bfloat16', 'decode-narrow-b17-bfloat16', 'prefill-2k-bfloat16', 'varlen-4x1k-bfloat16']}
def _integration700_sgl_gdn_register30_route(previous, keys):
    def route(case):
        if case.id in keys:
            from pathlib import Path
            import hashlib, sys
            package = Path('/opt/tileops-all-pairs/build/sgl-pristine-20261025T2310/source/python/sgl_kernel_npu/sgl_kernel_npu')
            binary = package / 'lib/libsgl_kernel_npu.so'
            if hashlib.sha256(binary.read_bytes()).hexdigest() != 'c8026d311e1bf56f8bff8a03a1908b420efc421c45980885ded6d69b51c24a6d':
                raise RuntimeError('SGL GDN verified library hash mismatch')
            candidate = _integration700_sgl_gdn_register30_build(case)
            if Path(sys.modules['sgl_kernel_npu'].__file__).resolve() != (package / '__init__.py').resolve():
                raise RuntimeError('SGL GDN resolved outside verified original package')
            return candidate
        if previous is not None:
            return previous(case)
        from benchmarks.ascend_baselines import UnsupportedCase
        raise UnsupportedCase('No SGL provider for this unregistered exact key')
    return route
for _integration700_sgl_gdn_register30_op, _integration700_sgl_gdn_register30_cases in _integration700_sgl_gdn_register30_keys.items():
    PROVIDERS[_integration700_sgl_gdn_register30_op] = _integration700_sgl_gdn_register30_route(PROVIDERS.get(_integration700_sgl_gdn_register30_op), frozenset(_integration700_sgl_gdn_register30_cases))
