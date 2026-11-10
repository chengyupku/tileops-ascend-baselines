"""Drain current batches before deploying streaming receipt finalization."""
import pathlib,json,subprocess,shlex,time,concurrent.futures
P=pathlib.Path(__file__).resolve().parent;R='/opt/torch-baseline1135-20261103';hosts=json.loads((P/'ASSIGNMENT.json').read_text())['hosts']
def ssh(h,cmd,**kw):return subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ClearAllForwardings=yes',hosts[h],cmd],check=True,**kw)
def roll(h):
 result=ssh(h,'python3 '+R+'/drain_at_job_boundary.py '+h,capture_output=True,text=True)
 old={'pid':'see DRAIN_BOUNDARY.json'}
 ssh(h,'rm '+R+'/'+h+'/STOP');subprocess.run(['systemctl','restart','torch-baseline1135-'+h+'.service'],check=True)
 (P/h/'SUPERVISOR_ROLLOUT.json').write_text(json.dumps({'at':time.time(),'prior_pid':old['pid'],'new_version':'streaming_per_job_receipts','completed':True},indent=2));print(h,'rolled',flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as p:list(p.map(roll,hosts))
