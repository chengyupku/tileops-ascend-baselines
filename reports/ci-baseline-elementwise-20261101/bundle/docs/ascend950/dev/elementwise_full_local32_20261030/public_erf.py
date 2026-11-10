"""Run original public tests selecting only Erf (serial development)."""
from pathlib import Path
import pytest
from tileops.kernels.ascend.elementwise import erf
ROOT=Path(__file__).resolve().parents[4]
class SelectErf:
 def pytest_collection_modifyitems(self,config,items):
  selected=[];removed=[]
  for item in items:
   values=getattr(getattr(item,'callspec',None),'params',{}).values()
   keep=(item.name.startswith('test_erf[') or item.name.startswith('test_erf_')) or any(getattr(v,'__name__','')=='ErfFwdOp' for v in values)
   (selected if keep else removed).append(item)
  items[:]=selected
  config.hook.pytest_deselected(items=removed)
raise SystemExit(pytest.main(['--tileops-device=npu',str(ROOT/'TileOPs/tests/ops/test_unary_math.py'),'-q','--junitxml='+str(Path(__file__).with_name('erf-public.xml'))],plugins=[SelectErf()]))
