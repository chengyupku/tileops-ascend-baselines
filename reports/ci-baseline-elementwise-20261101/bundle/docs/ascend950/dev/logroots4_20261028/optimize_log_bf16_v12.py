from pathlib import Path
import subprocess,sys
p=Path(__file__).resolve().parent
subprocess.run([sys.executable,str(p/'probe_log_bf16_v12.py')],check=True)
subprocess.run([sys.executable,str(p/'search_log_bf16_v12.py'),'--op','LogFwdOp','--case','16M-bfloat16','--output',str(p/'search/log-bf16-v12')],check=True)
