"""Bounded sequential candidate validation; invoked under one NPU guard."""
from pathlib import Path
import subprocess
import sys
HERE=Path(__file__).resolve().parent
subprocess.run([sys.executable,str(HERE/'probe_log1p_v5.py')],check=True)
for dtype in ('float16','float32'):
    subprocess.run([sys.executable,str(HERE/'search_log1p_v5.py'),'--op','Log1pFwdOp','--case','16M-'+dtype,'--output',str(HERE/'search'/('log1p-v5-'+dtype))],check=True)
