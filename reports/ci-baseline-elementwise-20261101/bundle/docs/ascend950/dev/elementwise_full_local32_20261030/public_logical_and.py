"""Run original public tests selecting only LogicalAnd (serial development)."""
from pathlib import Path
import pytest
from tileops.kernels.ascend.elementwise import logical_and
ROOT=Path(__file__).resolve().parents[4]
class SelectAnd:
 def pytest_collection_modifyitems(self,config,items):
  selected=[];removed=[]
  for item in items:
   values=getattr(getattr(item,'callspec',None),'params',{}).values()
   keep='logical_and' in item.name or any(getattr(v,'__name__','')=='LogicalAndFwdOp' for v in values)
   (selected if keep else removed).append(item)
  items[:]=selected
  config.hook.pytest_deselected(items=removed)
raise SystemExit(pytest.main(['--tileops-device=npu',str(ROOT/'TileOPs/tests/ops/test_logical.py'),'-q','--junitxml='+str(Path(__file__).with_name('logical-and-public.xml'))],plugins=[SelectAnd()]))
