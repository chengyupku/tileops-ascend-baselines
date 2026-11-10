"""Original native library + unchanged checker under bounded host guard."""
import pathlib,sys,json,subprocess,datetime,time,os
P=pathlib.Path(__file__).resolve().parent;parent=P.parent
task=P/'tasks'/sys.argv[1];t=json.loads(task.read_text());label=sys.argv[2]
cutoff=datetime.datetime(2026,11,9,17,52,tzinfo=datetime.timezone.utc).timestamp()
resume=parent/'RESUME_AFTER_NIGHTLY.json'
if resume.exists():
 r=json.loads(resume.read_text());assert r['nightly_completion_confirmed'] and r['remote_idle_confirmed']
 cutoff=datetime.datetime.fromisoformat(r['next_yield_utc'].replace('Z','+00:00')).timestamp()
if time.time()+60+900+60>=cutoff:raise SystemExit(75)
out=P/'runs'/t['id']/label;out.mkdir(parents=True,exist_ok=False)
env=dict(os.environ,LD_LIBRARY_PATH='/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64')
cmd=['python3','/opt/torch-v010-20261106/tools/npu_guard.py','run','--owner','root/baseline-overnight/native-'+t['id'],'--wait','60','--timeout','900','--receipt',str(out/'guard.json'),'--','bash',str(P/'launch.sh'),t['key'][0],str(P/'bootstrap.py'),'--task',str(task),'--out',str(out)]
with (out/'run.log').open('w') as log:r=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
print(json.dumps({'out':str(out),'key':t['key'],'exit':r.returncode}),flush=True);sys.exit(r.returncode)
