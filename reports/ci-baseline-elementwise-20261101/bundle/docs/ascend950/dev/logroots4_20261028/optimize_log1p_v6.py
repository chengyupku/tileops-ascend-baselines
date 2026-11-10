"""Bounded sequential candidate validation; invoked under one NPU guard."""
from pathlib import Path
import subprocess
import sys
HERE=Path(__file__).resolve().parent
subprocess.run([sys.executable,str(HERE/'probe_log1p_v6.py')],check=True)
for dtype in ('float16','bfloat16'):
    subprocess.run([sys.executable,str(HERE/'search_log1p_v6.py'),'--op','Log1pFwdOp','--case','16M-'+dtype,'--output',str(HERE/'search'/('log1p-v6-'+dtype))],check=True)
