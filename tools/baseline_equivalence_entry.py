"""Experiment entry selecting identical existing original NN staging on both hosts."""
from pathlib import Path
import runpy
ROOT=Path(__file__).resolve().parents[1]
original=runpy.run_path
def run_path(path,*args,**kwargs):
 if str(path)==str(ROOT/'tools/measure_baseline_cache_pair.py'):path=str(ROOT/'tools/measure_baseline_equivalence.py')
 return original(path,*args,**kwargs)
runpy.run_path=run_path
original(str(ROOT/'tools/baseline_old1076_nn_arm_staging_v3.py'),run_name='__main__')
