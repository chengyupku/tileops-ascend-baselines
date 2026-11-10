from pathlib import Path
import os
from .integration700_tla_sdk_recovery36_impl import factory as _factory
SDK=Path('/opt/tileops-all-pairs/workspace/.cache/all-pairs/remote_native_deepgemm_deepep/sdk-920b2')
ROOT=Path('/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/tla_sdk_recovery36')
KEYS=frozenset({('FP8QuantFwdOp', 'kv-index-8k-float16'), ('FP8QuantFwdOp', 'kv-index-context-chunk-float16'), ('FP8QuantFwdOp', 'kv-index-8k-bfloat16'), ('FP8QuantFwdOp', 'kv-index-context-chunk-bfloat16'), ('FP8QuantFwdOp', 'kv-index-4k-float32')})
def factory(case):
    assert (case._call.signature.name,case.id) in KEYS
    os.environ.update(BISHENG_HOME=str(SDK/'tools/bisheng_compiler'),ASCEND_HOME_PATH=str(SDK),ASCEND_TOOLKIT_HOME=str(SDK),CPATH=':'.join(str(SDK/x) for x in ['asc/include','asc','include']))
    op=case._call.signature.name
    out=ROOT/'provider_receipts'/op/case.id
    out.mkdir(parents=True,exist_ok=True)
    return _factory(case,op,out)
