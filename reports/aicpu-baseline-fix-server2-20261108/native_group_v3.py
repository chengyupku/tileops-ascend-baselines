"""Bounded foreground batch under one existing host guard; no nested guards."""
import pathlib,subprocess,json,sys,time,fcntl
P=pathlib.Path(__file__).resolve().parent;N=P/'native-v2'
with open('/run/lock/ascend-npu/device-0.lock') as f:
 try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:pass
 else:raise RuntimeError('host guard required')
rows=[]
for task in sorted((N/'runs').glob('*/a1/task.json')):
 if task.parent.parent.name in ['2de80f2b9bad4e01bb3d','2fea74995c281f088e0e']:continue
 if task.parent.parent.name=='54ad3ceb5b25198d019c':task=P/'native-v3/runs/54ad3ceb5b25198d019c/a1/task.json'
 out=task.parent;N=task.parents[3];assert not (out/'guard.json').exists() and not (out/'measurement.json').exists()
 print('START',json.load(open(task))['key'],flush=True);start=time.time()
 with (out/'run.log').open('w') as f:r=subprocess.run(['bash',str(N/'launch.sh'),'tilelang-ascend',str(N/'bootstrap.py'),'--task',str(task),'--out',str(out)],stdout=f,stderr=subprocess.STDOUT)
 row={'key':json.load(open(task))['key'],'out':str(out),'started_at':start,'finished_at':time.time(),'exit_code':r.returncode};rows.append(row);(P/'NATIVE_GROUP_V3_MEMBERS.json').write_text(json.dumps(rows,indent=2));print('EXIT',row['key'],r.returncode,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
 m=json.load(open(out/'measurement.json'))
 if any(w['profile_aicpu_task_count'] for w in m['timing']['retained_profiles']):
  (P/'NATIVE_GROUP_V3_FAILURE.json').write_text(json.dumps({'key':row['key'],'reason':'AICPU still present after fix'},indent=2));raise SystemExit(2)
