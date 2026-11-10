"""Use privately built unchanged DeepEP host sources with original CANN9.2."""
import os,sys,runpy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ['DEEPEP_ALL_PAIRS_BUILD']=str(ROOT/'reports/baseline-old1076-20261029/deepep_local_build/artifacts/host/lib')
sys.path[:0]=[str(ROOT/'TileOPs'),str(ROOT/'TileOPs/src'),str(ROOT)]
runpy.run_path(str(ROOT/'tools/measure_baseline_cache_pair.py'),run_name='__main__')
