"""Run unmodified Dropout public tests before development."""
from pathlib import Path
import pytest
from tileops.kernels.ascend.elementwise import dropout
ROOT=Path(__file__).resolve().parents[4]
raise SystemExit(pytest.main(['--tileops-device=npu',str(ROOT/'TileOPs/tests/ops/test_dropout.py'),'-q','--junitxml='+str(Path(__file__).with_name('dropout-public-control.xml'))]))
