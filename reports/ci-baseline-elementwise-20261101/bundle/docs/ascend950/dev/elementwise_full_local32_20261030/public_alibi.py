from pathlib import Path
import pytest
from tileops.kernels.ascend.elementwise import alibi
ROOT=Path(__file__).resolve().parents[4]
class Select:
 def pytest_collection_modifyitems(self,config,items):
  selected=[];removed=[]
  for item in items:
   (selected if 'alibi' in item.name else removed).append(item)
  items[:]=selected;config.hook.pytest_deselected(items=removed)
raise SystemExit(pytest.main(['--tileops-device=npu',str(ROOT/'TileOPs/tests/ops/test_special_elementwise.py'),'-q','--junitxml='+str(Path(__file__).with_name('alibi-public-v4.xml'))],plugins=[Select()]))
