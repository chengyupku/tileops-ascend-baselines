"""Original validated FlashAttentionScore; exact additional decode contracts."""
import json,os
from pathlib import Path
KEYS={'qwen3-30b-64k-float16','qwen3-30b-128k-float16'}
def gqa(case):
    assert case.id in KEYS and case._call.signature.name=='GQADenseFwdOp'
    m=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text())
    assert m['phase']=='running' and m['child']['pid']==os.getpgrp()
    from benchmarks.ascend_all_pairs.ops_transformer.validated_runtime import verify_pins,load
    verify_pins('attention')
    return load('ascend_transformer_graph').transformer_graph_gqa(case)
