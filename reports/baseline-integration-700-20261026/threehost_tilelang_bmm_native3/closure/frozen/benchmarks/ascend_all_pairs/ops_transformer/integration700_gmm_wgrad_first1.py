"""Canonical verified exact wgrad key using original complete GroupedMatmul V5."""
import importlib.util
import sys
from pathlib import Path

def verified_wgrad(case):
    assert case.id == 'moe-wgrad-float16' and case._call.signature.name == 'GroupedGemmFwdOp'
    path=Path('/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/gmm_wgrad_first1')
    def load(name, file):
        spec=importlib.util.spec_from_file_location(name,path/file)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    dependency=load('_gmm_wgrad_first1_v5','v5_fp32_bias.py')
    previous=sys.modules.get('v5_fp32_bias')
    sys.modules['v5_fp32_bias']=dependency
    try:binding=load('_gmm_wgrad_first1_binding','remaining_binding.py')
    finally:
        if previous is None:sys.modules.pop('v5_fp32_bias',None)
        else:sys.modules['v5_fp32_bias']=previous
    return binding.remaining(case)
