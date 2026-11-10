"""Reviewed exact Gemm keys; delegate unchanged tested adapter bytes."""
import hashlib
import os
from pathlib import Path
from types import ModuleType
_KEYS = frozenset([('GemmFwdOp', 'ds-v3-decode-qkv-a-bfloat16'), ('GemmFwdOp', 'ds-v3-m1-qkv-a-bfloat16'), ('GemmFwdOp', 'ds-v3-m16-mlp-up-bfloat16'), ('GemmFwdOp', 'ds-v3-m96-qkv-a-bfloat16'), ('GemmFwdOp', 'ds-v3-prefill-down-bfloat16'), ('GemmFwdOp', 'ds-v3-prefill-mlp-up-bfloat16'), ('GemmFwdOp', 'ds-v3-prefill-mlp-up-float16')])
_FROZEN = Path('/root/dyq/workspace/reports/baseline-integration-700-20261026/nn_registration02/adapter_037f902f6779aa1d84dad1d5d04777e6754cec5e9b3d653b7339ac5f0dd200d7.py')
_SHA256 = '037f902f6779aa1d84dad1d5d04777e6754cec5e9b3d653b7339ac5f0dd200d7'
_VENDOR = Path('/root/dyq/workspace/.cache/ascend-classic/native-pilot-20261020/ops-nn/install/vendors/classicpilot_nn')
def _load():
    data = _FROZEN.read_bytes()
    if hashlib.sha256(data).hexdigest() != _SHA256:
        raise RuntimeError('Reviewed frozen adapter changed')
    module = ModuleType('integration700_nn_registration02_frozen')
    module.__file__ = str(_FROZEN)
    exec(compile(data, str(_FROZEN), 'exec'), module.__dict__)
    return module

def make(case):
    from benchmarks.ascend_baselines import UnsupportedCase
    if (case._call.signature.name, case.id) not in _KEYS:
        raise UnsupportedCase('No nn_registration02 exact receipt')
    if Path(os.environ.get('OPS_NN_VENDOR', '')).resolve() != _VENDOR or os.environ.get('ASCEND_CUSTOM_OPP_PATH') != str(_VENDOR):
        raise UnsupportedCase('Select recorded classicpilot_nn vendor before runtime initialization: ' + str(_VENDOR))
    return _load().make(case)
