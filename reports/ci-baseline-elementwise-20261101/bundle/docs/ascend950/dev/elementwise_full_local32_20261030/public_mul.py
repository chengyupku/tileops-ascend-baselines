"""Original Mul public tests and Mul cases in shared broadcast/dtype tests."""
from pathlib import Path
import pytest
from tileops.kernels.ascend.elementwise import arithmetic_mul
ROOT=Path(__file__).resolve().parents[4]
class Select:
 def pytest_collection_modifyitems(self,config,items):
  chosen=[]
  for item in items:
   name=item.name.split('[')[0]
   params=getattr(getattr(item,'callspec',None),'params',{})
   selected=(name.startswith('test_mul_') or name in set() or getattr(params.get('op_cls'),'__name__','')=='MulFwdOp' or params.get('op_name')=='MulFwdOp')
   if selected:chosen.append(item)
  removed=[v for v in items if v not in chosen];items[:]=chosen
  config.hook.pytest_deselected(items=removed)
raise SystemExit(pytest.main(['--tileops-device=npu',str(ROOT/'TileOPs/tests/ops/test_binary_arith.py'),str(ROOT/'TileOPs/tests/ops/test_elementwise_binary_broadcast.py'),'-q','--junitxml='+str(Path(__file__).with_name('mul-public.xml'))],plugins=[Select()]))
