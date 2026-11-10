"""Run original public tests selecting only LogicalOr (serial development)."""
from pathlib import Path
import pytest
from tileops.kernels.ascend.elementwise import logical_or
ROOT=Path(__file__).resolve().parents[4]
class SelectOr:
 def pytest_collection_modifyitems(self,config,items):
  selected=[];removed=[]
  for item in items:
   values=getattr(getattr(item,'callspec',None),'params',{}).values()
   keep='logical_or' in item.name or any(getattr(v,'__name__','')=='LogicalOrFwdOp' for v in values)
   (selected if keep else removed).append(item)
  items[:]=selected
  config.hook.pytest_deselected(items=removed)
raise SystemExit(pytest.main(['--tileops-device=npu',str(ROOT/'TileOPs/tests/ops/test_logical.py'),'-q','--junitxml='+str(Path(__file__).with_name('logical-or-public.xml'))],plugins=[SelectOr()]))
