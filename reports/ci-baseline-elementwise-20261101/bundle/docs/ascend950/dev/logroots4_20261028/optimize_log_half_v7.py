from pathlib import Path
import subprocess
import sys
p=Path(__file__).resolve().parent
subprocess.run([sys.executable,str(p/'probe_log_half_v7.py')],check=True)
subprocess.run([sys.executable,str(p/'search_log_half_v7.py'),'--op','LogFwdOp','--case','16M-float16','--output',str(p/'search/log-half-v7')],check=True)
