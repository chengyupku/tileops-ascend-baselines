"""Original Mish public tests and inplace conformance."""
from pathlib import Path
import pytest
from tileops.kernels.ascend.elementwise import mish
ROOT=Path(__file__).resolve().parents[4]
class Select:
 def pytest_collection_modifyitems(self,config,items):
  selected=[];removed=[]
  for item in items:
   params=getattr(getattr(item,'callspec',None),'params',{})
   keep=(item.name.split('[')[0]=='test_mish' or item.name.startswith('test_mish_')) or params.get('op_name')=='MishFwdOp'
   (selected if keep else removed).append(item)
  items[:]=selected
  config.hook.pytest_deselected(items=removed)
raise SystemExit(pytest.main(['--tileops-device=npu',str(ROOT/'TileOPs/tests/ops/test_activation.py'),str(ROOT/'TileOPs/tests/ops/test_elementwise_unary_activation_conformance.py'),'-q','--junitxml='+str(Path(__file__).with_name('mish-public.xml'))],plugins=[Select()]))
