import sys,runpy
from pathlib import Path
ROOT=Path('/opt/tileops-baseline-cache/workspace')
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'TileOPs'),str(ROOT/'TileOPs/src'),str(ROOT)]
import baseline_old1076_transformer as old
from benchmarks.ascend_all_pairs import registry
registry.candidate_factory=old.factory
runpy.run_path(str(Path(__file__).parent/'bootstrap.py'),run_name='__main__')
