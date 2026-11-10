from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT/'TileOPs'),str(ROOT/'TileOPs/src'),str(ROOT)]
import torch
import torch_npu
def forbidden(*a,**kw):raise AssertionError('Event is forbidden')
torch.npu.Event=forbidden
import argparse
import importlib
p=argparse.ArgumentParser(add_help=False)
p.add_argument('--kernel-module',default='arithmetic')
a,rest=p.parse_known_args()
importlib.import_module('tileops.kernels.ascend.elementwise.'+a.kernel_module)
# The shared 16-bit test's name contains "div" even for its remainder parameter.
if a.kernel_module == 'arithmetic_div' and '-k' in rest:
    index = rest.index('-k') + 1
    rest[index] = '(' + rest[index] + ') and not remainder'
print('Effective pytest arguments:', rest, flush=True)
import pytest
raise SystemExit(pytest.main(['--tileops-device=npu',*rest]))
