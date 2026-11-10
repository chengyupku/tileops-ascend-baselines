"""Original ARM FLA package selection and lossless top-level gate transport."""
import sys,runpy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.cache/old1076/fla'))
runpy.run_path(str(ROOT/'tools/baseline_support_fla_v1.py'),run_name='__main__')
