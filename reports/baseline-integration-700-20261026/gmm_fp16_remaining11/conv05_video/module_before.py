"""Exact verified FP16 keys, original private GroupedMatmul V5 API."""
import importlib.util
import sys
from pathlib import Path

def conv1d_verified(case):
    assert case.id == 'whisper-conv1-float16'
    w=Path('/opt/tileops-all-pairs/workspace')
    d=w/'reports/baseline-integration-700-20261026/transformer_dense'
    path=w/'reports/baseline-integration-700-20261026/gmm_fp16_remaining11/conv02_fp32/conv1d_binding.py'
    spec=importlib.util.spec_from_file_location('_gmm_fp16_remaining11_tested_conv',path)
    m=importlib.util.module_from_spec(spec)
    sys.path.insert(0,str(d))
    try:spec.loader.exec_module(m)
    finally:sys.path.remove(str(d))
    return m.conv1d(case)

def verified_conv03_bias(case):
    assert case.id == 'whisper-conv1-bias-float16'
    path=Path('/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/gmm_fp16_remaining11/conv03_bias/remaining_binding.py')
    spec=importlib.util.spec_from_file_location('_gmm_fp16_remaining11_conv03_bias',path)
    m=importlib.util.module_from_spec(spec)
    sys.path.insert(0,str(path.parent))
    try:spec.loader.exec_module(m)
    finally:sys.path.remove(str(path.parent))
    return m.remaining(case)

def verified_hadamard01(case):
    assert case.id == 'llama-8b-prefill-float16'
    path=Path('/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/gmm_fp16_remaining11/hadamard01/remaining_binding.py')
    spec=importlib.util.spec_from_file_location('_gmm_fp16_remaining11_hadamard01',path)
    m=importlib.util.module_from_spec(spec)
    sys.path.insert(0,str(path.parent))
    try:spec.loader.exec_module(m)
    finally:sys.path.remove(str(path.parent))
    return m.remaining(case)

def verified_conv04_3d(case):
    assert case.id == 'unet-aspp-float16'
    path=Path('/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/gmm_fp16_remaining11/conv04_3d/remaining_binding.py')
    spec=importlib.util.spec_from_file_location('_gmm_fp16_remaining11_conv04_3d',path)
    m=importlib.util.module_from_spec(spec)
    sys.path.insert(0,str(path.parent))
    try:spec.loader.exec_module(m)
    finally:sys.path.remove(str(path.parent))
    return m.remaining(case)
