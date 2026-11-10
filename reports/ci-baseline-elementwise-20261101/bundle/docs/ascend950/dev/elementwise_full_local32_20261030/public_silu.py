"""Original SiLU public tests and inplace conformance."""
from pathlib import Path
import pytest
from tileops.kernels.ascend.elementwise import silu
ROOT=Path(__file__).resolve().parents[4]
class Select:
 def pytest_collection_modifyitems(self,config,items):
  selected=[];removed=[]
  for item in items:
   params=getattr(getattr(item,'callspec',None),'params',{})
   keep=(item.name.split('[')[0]=='test_silu' or item.name.startswith('test_silu_')) or params.get('op_name')=='SiluFwdOp'
   (selected if keep else removed).append(item)
  items[:]=selected
  config.hook.pytest_deselected(items=removed)
raise SystemExit(pytest.main(['--tileops-device=npu',str(ROOT/'TileOPs/tests/ops/test_activation.py'),str(ROOT/'TileOPs/tests/ops/test_elementwise_unary_activation_conformance.py'),'-q','--junitxml='+str(Path(__file__).with_name('silu-public.xml'))],plugins=[Select()]))
