"""Run original float and integer predicate tests for the current serial operator."""
import argparse,importlib
from pathlib import Path
import pytest
p=argparse.ArgumentParser();p.add_argument('kind',choices=['isnan','isinf','isfinite']);a=p.parse_args()
importlib.import_module('tileops.kernels.ascend.elementwise.'+a.kind)
root=Path(__file__).resolve().parents[4]
raise SystemExit(pytest.main(['--tileops-device=npu',str(root/'TileOPs/tests/ops/test_special_elementwise.py'),str(root/'TileOPs/tests/ops/test_unary_math.py'),'-q','-k',a.kind,'--junitxml='+str(Path(__file__).with_name(a.kind+'-public.xml'))]))
