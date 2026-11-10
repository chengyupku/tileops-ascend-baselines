"""Run original exponential public tests, excluding expm1."""
from pathlib import Path
import pytest
from tileops.kernels.ascend.elementwise import exp
ROOT=Path(__file__).resolve().parents[4]
raise SystemExit(pytest.main(['--tileops-device=npu',str(ROOT/'TileOPs/tests/ops/test_unary_math.py'),'-q','-k','test_exp and not test_expm1','--junitxml='+str(Path(__file__).with_name('exp-public.xml'))]))
