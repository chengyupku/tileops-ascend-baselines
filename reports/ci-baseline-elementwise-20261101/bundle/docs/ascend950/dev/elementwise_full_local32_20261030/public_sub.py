"""Original Sub public tests and Sub cases in shared broadcast/dtype tests."""
from pathlib import Path
import pytest
from tileops.kernels.ascend.elementwise import arithmetic_sub
ROOT=Path(__file__).resolve().parents[4]
class Select:
 def pytest_collection_modifyitems(self,config,items):
  chosen=[]
  for item in items:
   name=item.name.split('[')[0]
   params=getattr(getattr(item,'callspec',None),'params',{})
   selected=(name.startswith('test_sub_') or name in {'test_binary_op_rejects_runtime_dtype_mismatch'} or getattr(params.get('op_cls'),'__name__','')=='SubFwdOp' or params.get('op_name')=='SubFwdOp')
   if selected:chosen.append(item)
  removed=[v for v in items if v not in chosen];items[:]=chosen
  config.hook.pytest_deselected(items=removed)
raise SystemExit(pytest.main(['--tileops-device=npu',str(ROOT/'TileOPs/tests/ops/test_binary_arith.py'),str(ROOT/'TileOPs/tests/ops/test_elementwise_binary_broadcast.py'),'-q','--junitxml='+str(Path(__file__).with_name('sub-public.xml'))],plugins=[Select()]))
