"""Original MaskedFillScalar public tests, all dtype and broadcast combinations."""
from pathlib import Path
import pytest
from tileops.kernels.ascend.elementwise import masked_fill_scalar
ROOT=Path(__file__).resolve().parents[4]
class Select:
 def pytest_collection_modifyitems(self,config,items):
  chosen=[v for v in items if v.name.startswith('test_masked_fill') and not v.name.startswith('test_masked_fill_tensor_value')]
  removed=[v for v in items if v not in chosen];items[:]=chosen
  config.hook.pytest_deselected(items=removed)
raise SystemExit(pytest.main(['--tileops-device=npu',str(ROOT/'TileOPs/tests/ops/test_where_clamp_masked_fill_conformance.py'),str(ROOT/'TileOPs/tests/ops/test_special_elementwise.py'),str(ROOT/'TileOPs/tests/ops/test_multi_dtype_instance.py'),'-q','--junitxml='+str(Path(__file__).with_name('masked-fill-scalar-public.xml'))],plugins=[Select()]))
