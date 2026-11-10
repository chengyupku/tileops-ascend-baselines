"""One foreground attempt; bounded guard queue and hard pre-nightly admission."""
import datetime,json,os,pathlib,subprocess,sys,time
P=pathlib.Path(__file__).resolve().parent
task=P/'tasks'/sys.argv[1];label=sys.argv[2];mode=sys.argv[3] if len(sys.argv)>3 else 'probe'
t=json.loads(task.read_text());out=P/'runs'/str(t['ordinal'])/label
out.mkdir(parents=True,exist_ok=False)
cutoff=datetime.datetime(2026,11,9,17,52,tzinfo=datetime.timezone.utc).timestamp()
resume=P/'RESUME_AFTER_NIGHTLY.json'
if resume.exists():
 r=json.loads(resume.read_text());assert r['nightly_completion_confirmed'] and r['remote_idle_confirmed']
 cutoff=datetime.datetime.fromisoformat(r['next_yield_utc'].replace('Z','+00:00')).timestamp()
# Remote clock is behind the control host; cutoff includes >3 minutes margin.
def admitted():
 return time.time()+60+900+60<cutoff
if not admitted():
 (out/'DEFERRED.json').write_text(json.dumps({'reason':'nightly reservation','time':time.time()}));sys.exit(75)
with (out/'prepare.log').open('w') as log:
 r=subprocess.run(['bash',str(P/'launch_v5.sh'),'prepare',str(task),str(out)],stdout=log,stderr=subprocess.STDOUT)
if r.returncode:sys.exit(r.returncode)
if not admitted():
 (out/'DEFERRED.json').write_text(json.dumps({'reason':'nightly reservation after CPU prepare','time':time.time()}));sys.exit(75)
env=dict(os.environ,LD_LIBRARY_PATH='/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64')
cmd=['python3','/opt/torch-v010-20261106/tools/npu_guard.py','run','--owner','root/baseline-overnight/'+label,'--wait','60','--timeout','900','--receipt',str(out/'guard.json'),'--','bash',str(P/'launch_v5.sh'),mode,str(task),str(out)]
with (out/'device.log').open('w') as log:r=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
print(json.dumps({'out':str(out),'exit':r.returncode}),flush=True)
sys.exit(r.returncode)
