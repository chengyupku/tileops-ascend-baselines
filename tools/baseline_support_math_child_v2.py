"""Restore allowed original API compositions in the original foreground child."""
from functools import partial
from pathlib import Path
import runpy
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'TileOPs'), str(ROOT / 'TileOPs/src'), str(ROOT)]
from benchmarks.ascend_all_pairs.ops_math import NATIVE_CANDIDATES
from benchmarks.ascend_all_pairs.ops_math.composition_specs import COMPOSITIONS
from benchmarks.ascend_all_pairs.ops_math.composition import make
for op in COMPOSITIONS:
    NATIVE_CANDIDATES.setdefault(op, partial(make, op=op))
runpy.run_path(str(ROOT / 'TileOPs/benchmarks/ascend_all_pairs/ops_math/native_child.py'), run_name='__main__')
