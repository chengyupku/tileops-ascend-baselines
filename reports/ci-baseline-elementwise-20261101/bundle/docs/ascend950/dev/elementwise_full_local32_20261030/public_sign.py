"""Run original public tests selecting only Sign (serial development)."""
from pathlib import Path
import pytest
from tileops.kernels.ascend.elementwise import sign
ROOT=Path(__file__).resolve().parents[4]
class SelectSign:
 def pytest_collection_modifyitems(self,config,items):
  selected=[];removed=[]
  for item in items:
   values=getattr(getattr(item,'callspec',None),'params',{}).values()
   keep=item.name.startswith('test_sign') or any(getattr(v,'__name__','')=='SignFwdOp' for v in values)
   (selected if keep else removed).append(item)
  items[:]=selected
  config.hook.pytest_deselected(items=removed)
raise SystemExit(pytest.main(['--tileops-device=npu',str(ROOT/'TileOPs/tests/ops/test_unary_math.py'),'-q','--junitxml='+str(Path(__file__).with_name('sign-public.xml'))],plugins=[SelectSign()]))
