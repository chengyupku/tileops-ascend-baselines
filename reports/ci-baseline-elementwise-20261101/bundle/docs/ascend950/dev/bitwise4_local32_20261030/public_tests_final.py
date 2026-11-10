"""Run the unchanged public bitwise tests against the four Ascend entries."""
import importlib
from pathlib import Path
import sys
import torch
import torch_npu
root=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(root/'TileOPs'),str(root/'TileOPs/src'),str(root)]
for name in ['bitwise_binary','bitwise_or','bitwise_xor','bitwise_not']:
    importlib.import_module('tileops.kernels.ascend.elementwise.'+name)
import pytest
raise SystemExit(pytest.main(['--tileops-device=npu',str(root/'TileOPs/tests/ops/test_bitwise.py'),'-q','--junitxml='+str(root/'docs/ascend950/dev/bitwise4_local32_20261030/public-tests-final.xml')]))
