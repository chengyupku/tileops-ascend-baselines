from pathlib import Path
import pytest
from tileops.kernels.ascend.elementwise import nan_to_num
ROOT=Path(__file__).resolve().parents[4]
class Select:
 def pytest_collection_modifyitems(self,config,items):
  selected=[];removed=[]
  for item in items:
   (selected if 'nan_to_num' in item.name else removed).append(item)
  items[:]=selected;config.hook.pytest_deselected(items=removed)
raise SystemExit(pytest.main(['--tileops-device=npu',str(ROOT/'TileOPs/tests/ops/test_special_elementwise.py'),str(ROOT/'TileOPs/tests/ops/test_elementwise_unary_activation_conformance.py'),'-q','--junitxml='+str(Path(__file__).with_name('nan-to-num-public.xml'))],plugins=[Select()]))
