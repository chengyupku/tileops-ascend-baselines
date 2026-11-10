"""Original MaskedFillTensor public tests, all dtype and broadcast combinations."""
from pathlib import Path
import pytest
from tileops.kernels.ascend.elementwise import masked_fill_tensor
ROOT=Path(__file__).resolve().parents[4]
class Select:
 def pytest_collection_modifyitems(self,config,items):
  chosen=[v for v in items if v.name.startswith('test_masked_fill_tensor_value')]
  removed=[v for v in items if v not in chosen];items[:]=chosen
  config.hook.pytest_deselected(items=removed)
raise SystemExit(pytest.main(['--tileops-device=npu',str(ROOT/'TileOPs/tests/ops/test_where_clamp_masked_fill_conformance.py'),'-q','--junitxml='+str(Path(__file__).with_name('masked-fill-tensor-public.xml'))],plugins=[Select()]))
