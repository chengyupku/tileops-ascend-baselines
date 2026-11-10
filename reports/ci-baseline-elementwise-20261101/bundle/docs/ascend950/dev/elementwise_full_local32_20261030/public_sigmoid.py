"""Run original public tests selecting only Sigmoid (serial development)."""
from pathlib import Path
import pytest
from tileops.kernels.ascend.elementwise import sigmoid
ROOT=Path(__file__).resolve().parents[4]
class SelectSigmoid:
 def pytest_collection_modifyitems(self,config,items):
  selected=[];removed=[]
  for item in items:
   values=getattr(getattr(item,'callspec',None),'params',{}).values()
   keep=(item.name.startswith('test_sigmoid[') or item.name.startswith('test_sigmoid_') or item.name.startswith('test_16bit_sigmoid_')) or any(getattr(v,'__name__','')=='SigmoidFwdOp' for v in values)
   (selected if keep else removed).append(item)
  items[:]=selected
  config.hook.pytest_deselected(items=removed)
raise SystemExit(pytest.main(['--tileops-device=npu',str(ROOT/'TileOPs/tests/ops/test_activation.py'),'-q','--junitxml='+str(Path(__file__).with_name('sigmoid-public.xml'))],plugins=[SelectSigmoid()]))
