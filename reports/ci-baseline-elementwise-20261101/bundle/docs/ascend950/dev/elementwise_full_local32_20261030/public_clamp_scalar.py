from pathlib import Path
import pytest
from tileops.kernels.ascend.elementwise import clamp_scalar
ROOT=Path(__file__).resolve().parents[4]
class Select:
 def pytest_collection_modifyitems(self,config,items):
  selected=[];removed=[]
  for item in items:
   keep='clamp_scalar' in item.name
   (selected if keep else removed).append(item)
  items[:]=selected;config.hook.pytest_deselected(items=removed)
raise SystemExit(pytest.main(['--tileops-device=npu',str(ROOT/'TileOPs/tests/ops/test_where_clamp_masked_fill_conformance.py'),'-q','--junitxml='+str(Path(__file__).with_name('clamp-scalar-public.xml'))],plugins=[Select()]))
