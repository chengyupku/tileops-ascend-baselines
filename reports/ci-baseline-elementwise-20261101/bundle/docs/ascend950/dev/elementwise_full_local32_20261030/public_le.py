"""Original Le public tests and Le cases in shared broadcast/dtype tests."""
from pathlib import Path
import pytest
import functools
from tileops.kernels.ascend.elementwise import comparison
ROOT=Path(__file__).resolve().parents[4]
class Select:
 def pytest_collection_modifyitems(self,config,items):
  chosen=[]
  for item in items:
   name=item.name.split('[')[0]
   params=getattr(getattr(item,'callspec',None),'params',{})
   maker=params.get('make_op')
   if isinstance(maker,functools.partial):maker=maker.func
   selected=(name.startswith('test_le_') or name == 'test_comparison_nan_ordering' or getattr(maker,'__name__','')=='LeFwdOp' or getattr(params.get('op_cls'),'__name__','')=='LeFwdOp' or params.get('op_name')=='LeFwdOp')
   if selected and item.get_closest_marker('cuda_only') is None:chosen.append(item)
  removed=[v for v in items if v not in chosen];items[:]=chosen
  config.hook.pytest_deselected(items=removed)
raise SystemExit(pytest.main(['--tileops-device=npu',str(ROOT/'TileOPs/tests/ops/test_comparison.py'),str(ROOT/'TileOPs/tests/ops/test_elementwise_binary_broadcast.py'),'-q','--junitxml='+str(Path(__file__).with_name('le-public.xml'))],plugins=[Select()]))
