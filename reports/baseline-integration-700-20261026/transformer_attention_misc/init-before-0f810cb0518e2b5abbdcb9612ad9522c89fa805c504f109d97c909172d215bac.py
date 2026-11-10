"""Isolated ops-transformer providers; importing this module is device-free."""
from .adapters import bmm

PROVIDERS = {'BmmFwdOp': bmm}

# Server2 exact-key integrations. Imports are device-free; factories initialize
# original runtimes only when an accepted case is actually executed.
from .attention_adapter import gqa as _server2_gqa
from .gdn_adapter import gdn as _server2_gdn
PROVIDERS['GQADenseFwdOp'] = _server2_gqa
PROVIDERS['GDNFwdOp'] = _server2_gdn

# integration700_transformer_linear exact GDN keys
from .integration700_transformer_linear_gdn import wrap as _integration700_transformer_linear_wrap
PROVIDERS["GDNFwdOp"] = _integration700_transformer_linear_wrap(PROVIDERS.get("GDNFwdOp"))

# integration700 transformer_attention_misc: complete canonical passes only
from .integration700_transformer_attention_misc_dense import gqa as _i700_misc_GQADenseFwdOp
def _i700_misc_GQADenseFwdOp_route(case, previous=PROVIDERS.get('GQADenseFwdOp')):
    if case.id in {'qwen3-30b-64k-float16', 'qwen3-30b-128k-float16'}:
        return _i700_misc_GQADenseFwdOp(case)
    if previous is not None:
        return previous(case)
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase("No registered factory for this exact key")
PROVIDERS['GQADenseFwdOp'] = _i700_misc_GQADenseFwdOp_route
from .integration700_transformer_attention_misc_decode import factory as _i700_misc_MHADecodePagedWithKVCacheFwdOp
def _i700_misc_MHADecodePagedWithKVCacheFwdOp_route(case, previous=PROVIDERS.get('MHADecodePagedWithKVCacheFwdOp')):
    if case.id in {'serving-b2-float16'}:
        return _i700_misc_MHADecodePagedWithKVCacheFwdOp(case)
    if previous is not None:
        return previous(case)
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase("No registered factory for this exact key")
PROVIDERS['MHADecodePagedWithKVCacheFwdOp'] = _i700_misc_MHADecodePagedWithKVCacheFwdOp_route

# integration700 transformer_attention_misc: complete canonical passes only
from .integration700_transformer_attention_misc_dense import gqa as _i700_misc_GQADenseFwdOp
def _i700_misc_GQADenseFwdOp_route(case, previous=PROVIDERS.get('GQADenseFwdOp')):
    if case.id in {'qwen3-30b-128k-float16', 'qwen3-30b-64k-float16'}:
        return _i700_misc_GQADenseFwdOp(case)
    if previous is not None:
        return previous(case)
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase("No registered factory for this exact key")
PROVIDERS['GQADenseFwdOp'] = _i700_misc_GQADenseFwdOp_route
from .integration700_transformer_attention_misc_decode import factory as _i700_misc_MHADecodePagedWithKVCacheFwdOp
def _i700_misc_MHADecodePagedWithKVCacheFwdOp_route(case, previous=PROVIDERS.get('MHADecodePagedWithKVCacheFwdOp')):
    if case.id in {'serving-kv512-float16', 'speculative-decode-float16', 'serving-p128-float16', 'serving-kv1k-float16', 'serving-b2-float16'}:
        return _i700_misc_MHADecodePagedWithKVCacheFwdOp(case)
    if previous is not None:
        return previous(case)
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase("No registered factory for this exact key")
PROVIDERS['MHADecodePagedWithKVCacheFwdOp'] = _i700_misc_MHADecodePagedWithKVCacheFwdOp_route
from .integration700_transformer_attention_misc_decode import factory as _i700_misc_MLADecodeWithKVCacheFwdOp
def _i700_misc_MLADecodeWithKVCacheFwdOp_route(case, previous=PROVIDERS.get('MLADecodeWithKVCacheFwdOp')):
    if case.id in {'mla-tp-b1-float16'}:
        return _i700_misc_MLADecodeWithKVCacheFwdOp(case)
    if previous is not None:
        return previous(case)
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase("No registered factory for this exact key")
PROVIDERS['MLADecodeWithKVCacheFwdOp'] = _i700_misc_MLADecodeWithKVCacheFwdOp_route

# integration700 transformer_attention_misc: complete canonical passes only
from .integration700_transformer_attention_misc_dense import gqa as _i700_misc_GQADenseFwdOp
def _i700_misc_GQADenseFwdOp_route(case, previous=PROVIDERS.get('GQADenseFwdOp')):
    if case.id in {'qwen3-30b-128k-float16', 'qwen3-30b-64k-float16'}:
        return _i700_misc_GQADenseFwdOp(case)
    if previous is not None:
        return previous(case)
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase("No registered factory for this exact key")
PROVIDERS['GQADenseFwdOp'] = _i700_misc_GQADenseFwdOp_route
from .integration700_transformer_attention_misc_decode import factory as _i700_misc_MHADecodePagedWithKVCacheFwdOp
def _i700_misc_MHADecodePagedWithKVCacheFwdOp_route(case, previous=PROVIDERS.get('MHADecodePagedWithKVCacheFwdOp')):
    if case.id in {'serving-kv512-float16', 'serving-kv1k-float16', 'serving-p128-float16', 'speculative-decode-float16', 'serving-b2-float16'}:
        return _i700_misc_MHADecodePagedWithKVCacheFwdOp(case)
    if previous is not None:
        return previous(case)
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase("No registered factory for this exact key")
PROVIDERS['MHADecodePagedWithKVCacheFwdOp'] = _i700_misc_MHADecodePagedWithKVCacheFwdOp_route
from .integration700_transformer_attention_misc_decode import factory as _i700_misc_MLADecodeWithKVCacheFwdOp
def _i700_misc_MLADecodeWithKVCacheFwdOp_route(case, previous=PROVIDERS.get('MLADecodeWithKVCacheFwdOp')):
    if case.id in {'mla-tp-b1-float16'}:
        return _i700_misc_MLADecodeWithKVCacheFwdOp(case)
    if previous is not None:
        return previous(case)
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase("No registered factory for this exact key")
PROVIDERS['MLADecodeWithKVCacheFwdOp'] = _i700_misc_MLADecodeWithKVCacheFwdOp_route

# integration700 transformer_attention_misc: complete canonical passes only
from .integration700_transformer_attention_misc_dense import gqa as _i700_misc_GQADenseFwdOp
def _i700_misc_GQADenseFwdOp_route(case, previous=PROVIDERS.get('GQADenseFwdOp')):
    if case.id in {'qwen3-30b-128k-float16', 'qwen3-30b-64k-float16'}:
        return _i700_misc_GQADenseFwdOp(case)
    if previous is not None:
        return previous(case)
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase("No registered factory for this exact key")
PROVIDERS['GQADenseFwdOp'] = _i700_misc_GQADenseFwdOp_route
from .integration700_transformer_attention_misc_decode import factory as _i700_misc_MHADecodePagedWithKVCacheFwdOp
def _i700_misc_MHADecodePagedWithKVCacheFwdOp_route(case, previous=PROVIDERS.get('MHADecodePagedWithKVCacheFwdOp')):
    if case.id in {'serving-kv1k-float16', 'serving-p128-float16', 'speculative-decode-float16', 'serving-b2-float16', 'serving-kv512-float16'}:
        return _i700_misc_MHADecodePagedWithKVCacheFwdOp(case)
    if previous is not None:
        return previous(case)
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase("No registered factory for this exact key")
PROVIDERS['MHADecodePagedWithKVCacheFwdOp'] = _i700_misc_MHADecodePagedWithKVCacheFwdOp_route
from .integration700_transformer_attention_misc_decode import factory as _i700_misc_MLADecodeWithKVCacheFwdOp
def _i700_misc_MLADecodeWithKVCacheFwdOp_route(case, previous=PROVIDERS.get('MLADecodeWithKVCacheFwdOp')):
    if case.id in {'mla-b128-4096-bfloat16', 'mla-b128-4096-float16', 'mla-tp-b1-float16', 'mla-b128-8192-float16', 'mla-b128-8192-bfloat16'}:
        return _i700_misc_MLADecodeWithKVCacheFwdOp(case)
    if previous is not None:
        return previous(case)
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase("No registered factory for this exact key")
PROVIDERS['MLADecodeWithKVCacheFwdOp'] = _i700_misc_MLADecodeWithKVCacheFwdOp_route

# integration700 transformer_attention_misc: complete canonical passes only
from .integration700_transformer_attention_misc_dense import gqa as _i700_misc_GQADenseFwdOp
def _i700_misc_GQADenseFwdOp_route(case, previous=PROVIDERS.get('GQADenseFwdOp')):
    if case.id in {'qwen3-30b-128k-float16', 'qwen3-30b-64k-float16'}:
        return _i700_misc_GQADenseFwdOp(case)
    if previous is not None:
        return previous(case)
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase("No registered factory for this exact key")
PROVIDERS['GQADenseFwdOp'] = _i700_misc_GQADenseFwdOp_route
from .integration700_transformer_attention_misc_decode import factory as _i700_misc_MHADecodePagedWithKVCacheFwdOp
def _i700_misc_MHADecodePagedWithKVCacheFwdOp_route(case, previous=PROVIDERS.get('MHADecodePagedWithKVCacheFwdOp')):
    if case.id in {'serving-kv1k-float16', 'speculative-decode-float16', 'serving-p128-float16', 'serving-kv512-float16', 'serving-b2-float16'}:
        return _i700_misc_MHADecodePagedWithKVCacheFwdOp(case)
    if previous is not None:
        return previous(case)
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase("No registered factory for this exact key")
PROVIDERS['MHADecodePagedWithKVCacheFwdOp'] = _i700_misc_MHADecodePagedWithKVCacheFwdOp_route
from .integration700_transformer_attention_misc_decode import factory as _i700_misc_MLADecodeWithKVCacheFwdOp
def _i700_misc_MLADecodeWithKVCacheFwdOp_route(case, previous=PROVIDERS.get('MLADecodeWithKVCacheFwdOp')):
    if case.id in {'mla-b128-4096-bfloat16', 'mla-b128-32768-float16', 'mla-b128-8192-float16', 'mla-b128-16384-float16', 'mla-b128-16384-bfloat16', 'mla-b128-4096-float16', 'mla-b128-8192-bfloat16', 'mla-b128-32768-bfloat16', 'mla-tp-b1-float16'}:
        return _i700_misc_MLADecodeWithKVCacheFwdOp(case)
    if previous is not None:
        return previous(case)
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase("No registered factory for this exact key")
PROVIDERS['MLADecodeWithKVCacheFwdOp'] = _i700_misc_MLADecodeWithKVCacheFwdOp_route
from .integration700_transformer_attention_misc_dsa import factory as _i700_misc_DSADecodeWithKVCacheFwdOp
def _i700_misc_DSADecodeWithKVCacheFwdOp_route(case, previous=PROVIDERS.get('DSADecodeWithKVCacheFwdOp')):
    if case.id in {'ds-v32-kv2k-bfloat16', 'mla-d512-h64-kv64k-bfloat16', 'ds-v32-kv4k-float16', 'mla-d512-h128-kv32k-bfloat16', 'mla-d512-h128-kv48k-bfloat16', 'ds-v32-kv2k-float16', 'ds-v32-kv4k-bfloat16', 'mla-d512-h64-kv48k-bfloat16'}:
        return _i700_misc_DSADecodeWithKVCacheFwdOp(case)
    if previous is not None:
        return previous(case)
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase("No registered factory for this exact key")
PROVIDERS['DSADecodeWithKVCacheFwdOp'] = _i700_misc_DSADecodeWithKVCacheFwdOp_route
