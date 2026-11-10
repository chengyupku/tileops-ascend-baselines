"""Run original public tests selecting only Sin (serial development)."""
from pathlib import Path
import pytest
from tileops.kernels.ascend.elementwise import sin
ROOT=Path(__file__).resolve().parents[4]
class SelectSin:
 def pytest_collection_modifyitems(self,config,items):
  selected=[];removed=[]
  for item in items:
   values=getattr(getattr(item,'callspec',None),'params',{}).values()
   keep=(item.name.startswith('test_sin[') or item.name.startswith('test_sin_')) or any(getattr(v,'__name__','')=='SinFwdOp' for v in values)
   (selected if keep else removed).append(item)
  items[:]=selected
  config.hook.pytest_deselected(items=removed)
raise SystemExit(pytest.main(['--tileops-device=npu',str(ROOT/'TileOPs/tests/ops/test_unary_math.py'),'-q','--junitxml='+str(Path(__file__).with_name('sin-public.xml'))],plugins=[SelectSin()]))
