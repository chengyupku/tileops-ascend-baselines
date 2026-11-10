"""Run original public tests selecting only Reciprocal (serial development)."""
from pathlib import Path
import pytest
from tileops.kernels.ascend.elementwise import reciprocal
ROOT=Path(__file__).resolve().parents[4]
class SelectReciprocal:
 def pytest_collection_modifyitems(self,config,items):
  selected=[];removed=[]
  for item in items:
   values=getattr(getattr(item,'callspec',None),'params',{}).values()
   keep='reciprocal' in item.name or any(getattr(v,'__name__','')=='ReciprocalFwdOp' for v in values)
   (selected if keep else removed).append(item)
  items[:]=selected
  config.hook.pytest_deselected(items=removed)
raise SystemExit(pytest.main(['--tileops-device=npu',str(ROOT/'TileOPs/tests/ops/test_unary_math.py'),'-q','--junitxml='+str(Path(__file__).with_name('reciprocal-public.xml'))],plugins=[SelectReciprocal()]))
