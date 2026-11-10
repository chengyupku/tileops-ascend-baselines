from pathlib import Path
import pytest
from tileops.kernels.ascend.elementwise import lerp
ROOT=Path(__file__).resolve().parents[4]
class Select:
 def pytest_collection_modifyitems(self,config,items):
  selected=[];removed=[]
  for item in items:
   (selected if 'lerp_tensor' in item.name else removed).append(item)
  items[:]=selected;config.hook.pytest_deselected(items=removed)
raise SystemExit(pytest.main(['--tileops-device=npu',str(ROOT/'TileOPs/tests/ops/test_binary_arith.py'),'-q','--junitxml='+str(Path(__file__).with_name('lerp-tensor-public-v1.xml'))],plugins=[Select()]))
