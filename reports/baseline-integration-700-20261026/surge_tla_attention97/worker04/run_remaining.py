from pathlib import Path
import subprocess,sys,json
P=Path(__file__).resolve().parent
cases=['qwen3-30b-4k-float16','qwen3-30b-16k-float16','qwen3-30b-32k-float16','qwen3-30b-64k-float16','qwen3-30b-128k-float16','qwen3-30b-256k-float16']
for cid in cases:
 if (P/'attempts'/cid/'guard.json').exists():continue
 p=subprocess.run([sys.executable,str(P/'orchestrate.py'),cid]);print('CASE_FINISHED',cid,p.returncode,flush=True)
 subprocess.run([sys.executable,str(P/'summarize.py')],check=True)
