"""Serial foreground reservations; preserve failures and stop for inspection."""
import pathlib,subprocess,os,json,time
P=pathlib.Path(__file__).resolve().parent/'native-v2'
os.environ['LD_LIBRARY_PATH']='/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64'
for task in sorted((P/'runs').glob('*/a1/task.json')):
 if (P/'STOP').exists():raise SystemExit('own batch stopped before next reservation')
 out=task.parent
 if (out/'guard.json').exists():raise RuntimeError('refuse overwrite '+str(out))
 print('START',json.load(open(task))['key'],flush=True)
 cmd=['python3','/opt/torch-v010-20261106/tools/npu_guard.py','run','--owner','root/aicpu-fix-server2/native-v2/'+task.parent.parent.name,'--wait','600','--timeout','600','--receipt',str(out/'guard.json'),'--','bash',str(P/'launch.sh'),'tilelang-ascend',str(P/'bootstrap.py'),'--task',str(task),'--out',str(out)]
 with (out/'run.log').open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 print('EXIT',task.parent.parent.name,r.returncode,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
