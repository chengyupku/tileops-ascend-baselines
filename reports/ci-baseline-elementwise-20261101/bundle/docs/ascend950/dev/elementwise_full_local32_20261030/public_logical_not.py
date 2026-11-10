"""Run the unchanged public logical NOT tests."""
from pathlib import Path
import pytest
from tileops.kernels.ascend.elementwise import logical_not
ROOT=Path(__file__).resolve().parents[4]
raise SystemExit(pytest.main(['--tileops-device=npu',str(ROOT/'TileOPs/tests/ops/test_logical.py'),'-q','-k','test_logical_not','--junitxml='+str(Path(__file__).with_name('logical-not-public.xml'))]))
