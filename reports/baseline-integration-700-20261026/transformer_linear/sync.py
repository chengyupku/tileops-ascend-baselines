from pathlib import Path
import subprocess,tarfile,io
W=Path(__file__).resolve().parent;remote='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/transformer_linear';ssh=['/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh','openlibing-DevEnv-055409']
feedback=W/'FEEDBACK.txt'
if feedback.exists():subprocess.run(ssh+['cat > '+remote+'/FEEDBACK.txt'],input=feedback.read_bytes(),check=True)
subprocess.run(ssh+['python3 '+remote+'/heartbeat_tick.py'],check=True)
# Only receipts and scripts, no device/cache/compiler trees.
cmd="cd "+remote+" && tar -czf - --exclude='./*/TMPDIR' --exclude='./*/XDG*' --exclude='./*/ASCEND*' --exclude='./*/TORCH*' --exclude='./*/TRITON*' --exclude='./*/PIP*' --exclude='./*/CCACHE*' --exclude='*.pyc' --exclude='./FEEDBACK.txt' ."
p=subprocess.run(ssh+[cmd],stdout=subprocess.PIPE,check=False)
if p.returncode not in (0,1):raise RuntimeError('SSH/tar sync failed: '+str(p.returncode))
with tarfile.open(fileobj=io.BytesIO(p.stdout),mode='r:gz') as t:t.extractall(W,filter='data')
