"""Shared six-comparison public regression after Eq SIMD bias index repair."""
from pathlib import Path
import pytest
from tileops.kernels.ascend.elementwise import comparison
ROOT=Path(__file__).resolve().parents[4]
class Select:
 def pytest_collection_modifyitems(self,config,items):
  removed=[v for v in items if v.get_closest_marker('cuda_only') is not None]
  items[:]=[v for v in items if v not in removed]
  config.hook.pytest_deselected(items=removed)
raise SystemExit(pytest.main(['--tileops-device=npu',str(ROOT/'TileOPs/tests/ops/test_comparison.py'),'-q','--junitxml='+str(Path(__file__).with_name('eq-shared-comparison-public.xml'))],plugins=[Select()]))
