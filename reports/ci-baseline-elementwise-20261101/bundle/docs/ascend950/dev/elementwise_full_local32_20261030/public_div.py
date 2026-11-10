"""Original Div public tests and Div cases in shared broadcast/dtype tests."""
from pathlib import Path
import pytest
import functools
from tileops.kernels.ascend.elementwise import arithmetic_div, arithmetic_modulo
ROOT=Path(__file__).resolve().parents[4]
class Select:
 def pytest_collection_modifyitems(self,config,items):
  chosen=[]
  for item in items:
   name=item.name.split('[')[0]
   params=getattr(getattr(item,'callspec',None),'params',{})
   maker=params.get('make_op')
   if isinstance(maker,functools.partial):maker=maker.func
   selected=(name.startswith('test_div_') or getattr(maker,'__name__','')=='DivFwdOp' or getattr(params.get('op_cls'),'__name__','')=='DivFwdOp' or params.get('op_name')=='DivFwdOp')
   if selected:chosen.append(item)
  removed=[v for v in items if v not in chosen];items[:]=chosen
  config.hook.pytest_deselected(items=removed)
raise SystemExit(pytest.main(['--tileops-device=npu',str(ROOT/'TileOPs/tests/ops/test_binary_arith.py'),str(ROOT/'TileOPs/tests/ops/test_elementwise_binary_broadcast.py'),'-q','--junitxml='+str(Path(__file__).with_name('div-public.xml'))],plugins=[Select()]))
