import argparse,importlib
from pathlib import Path
import pytest
p=argparse.ArgumentParser();p.add_argument('kind',choices=['log','log1p','sqrt','rsqrt']);a=p.parse_args()
importlib.import_module('tileops.kernels.ascend.elementwise.'+a.kind)
root=Path(__file__).resolve().parents[4]
selection='test_'+a.kind
if a.kind=='log':selection+=' and not test_log1p'
raise SystemExit(pytest.main(['--tileops-device=npu',str(root/'TileOPs/tests/ops/test_unary_math.py'),'-q','-k',selection,'--junitxml='+str(Path(__file__).with_name(a.kind+'-public.xml'))]))
