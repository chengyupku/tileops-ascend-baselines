from pathlib import Path
import pytest
from tileops.kernels.ascend.elementwise import gelu_tanh_and_mul
ROOT=Path(__file__).resolve().parents[4]
class Select:
 def pytest_collection_modifyitems(self,config,items):
  selected=[];removed=[]
  for item in items:
   (selected if 'gelu_tanh_and_mul' in item.name else removed).append(item)
  items[:]=selected;config.hook.pytest_deselected(items=removed)
raise SystemExit(pytest.main(['--tileops-device=npu',str(ROOT/'TileOPs/tests/ops/test_fused_gated.py'),'-q','--junitxml='+str(Path(__file__).with_name('gelu-tanh-and-mul-public-v2.xml'))],plugins=[Select()]))
