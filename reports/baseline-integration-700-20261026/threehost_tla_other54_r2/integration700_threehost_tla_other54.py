"""Reviewable exact-key binding proposal; original full APIs, official SDK headers.

No registration is performed by importing this module. Use the original TileLang
0.1.15 ARM runtime from the validated deployment, under the host NPU guard.
"""
from pathlib import Path
import os
import importlib.util

KEYS = {
 ('BmmFwdOp', 'mha-decode-pv-bfloat16'),
 ('BmmFwdOp', 'mha-decode-pv-float16'),
 ('BmmFwdOp', 'moe-prefill-bfloat16'),
 ('GQADenseFwdOp', 'falcon-7b-2k-bfloat16'),
 ('GQADenseFwdOp', 'llama-8b-4k-bfloat16'),
 ('MoEGroupedGemmFwdOp', 'qwen3-30b-ep8-decode-bfloat16'),
}

def factory(case, op):
    if (op, case.id) not in KEYS:
        raise ValueError('Exact key outside reviewed scope')
    w = Path('/opt/tileops-all-pairs/workspace')
    p = w / 'reports/baseline-integration-700-20261026/threehost_tla_other54_r2'
    sdk = w / '.cache/all-pairs/remote_native_deepgemm_deepep/sdk-920-release-headers'
    os.environ.update(BISHENG_HOME=str(sdk / 'tools/bisheng_compiler'),
                      ASCEND_HOME_PATH=str(sdk), ASCEND_TOOLKIT_HOME=str(sdk),
                      CPATH=':'.join(str(sdk / x) for x in ['asc/include', 'asc', 'include']))
    import tilelang
    assert Path(tilelang.__file__).resolve() == Path('/opt/tileops-ci/runtime/lib/python3.12/site-packages/tilelang/__init__.py')
    spec = importlib.util.spec_from_file_location('_original_cube_binding_54', p / 'tilelang_adapter.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = p / 'parent_review_runs' / op
    out.mkdir(parents=True, exist_ok=True)
    return mod.factory(case, op, out)
