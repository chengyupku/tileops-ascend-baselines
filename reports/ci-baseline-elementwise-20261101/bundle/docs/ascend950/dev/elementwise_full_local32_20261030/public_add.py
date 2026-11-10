"""Original Add public tests and Add cases in shared broadcast/dtype tests."""
from pathlib import Path
import pytest
from tileops.kernels.ascend.elementwise import arithmetic
ROOT=Path(__file__).resolve().parents[4]
class Select:
 def pytest_collection_modifyitems(self,config,items):
  chosen=[]
  for item in items:
   name=item.name.split('[')[0]
   params=getattr(getattr(item,'callspec',None),'params',{})
   selected=(name.startswith('test_add_') or name in {'test_binary_arith_integer_dtype_add','test_binary_op_does_not_keep_its_inputs','test_row_broadcast_tail_matches_torch'} or getattr(params.get('op_cls'),'__name__','')=='AddFwdOp' or params.get('op_name')=='AddFwdOp')
   if selected:chosen.append(item)
  removed=[v for v in items if v not in chosen];items[:]=chosen
  config.hook.pytest_deselected(items=removed)
raise SystemExit(pytest.main(['--tileops-device=npu',str(ROOT/'TileOPs/tests/ops/test_binary_arith.py'),str(ROOT/'TileOPs/tests/ops/test_elementwise_binary_broadcast.py'),'-q','--junitxml='+str(Path(__file__).with_name('add-public.xml'))],plugins=[Select()]))
