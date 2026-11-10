import pathlib,subprocess,json,shlex,datetime,time,sys
P=pathlib.Path(__file__).resolve().parent;SSH='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh';REMOTE='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/'+P.name;HOST='openlibing-DevEnv-661495'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def heart(phase,key): (P/'HEARTBEAT.json').write_text(json.dumps(dict(utc=now(),phase=phase,key=key,accepted_added=0),indent=2))
def remote(cmd,log):
 with (P/log).open('x') as f:
  proc=subprocess.Popen([SSH,HOST,shlex.join(cmd)],stdout=f,stderr=subprocess.STDOUT)
  while True:
   try:return proc.wait(timeout=20)
   except subprocess.TimeoutExpired:heart('foreground_'+log,KEY)
for row in json.loads((P/'SCOPE.json').read_text()):
 KEY=[row['baseline'],row['operator'],row['case_id']];tag='_'.join(KEY);heart('pre_hash',KEY)
 rc=remote(['python3',REMOTE+'/freeze.py','pre_'+tag],tag+'.pre.log');assert rc==0
 cmd=['npu-guard','run','--owner','baseline700-20261026/threehost_tilelang3_complete_capture','--wait','120','--timeout','180','--','/opt/tileops-all-pairs/bin/allpairs-python',REMOTE+'/run_case.py',*KEY]
 rec=dict(key=KEY,host=HOST,argv=cmd,started=now(),timeout_rationale='Prior full checker 15-19 seconds; 180 seconds sufficient.');heart('guard',KEY)
 rc=remote(cmd,tag+'.guard.log');rec.update(finished=now(),returncode=rc);(P/(tag+'.guard.json')).write_text(json.dumps(rec,indent=2))
 heart('post_hash',KEY);post=remote(['python3',REMOTE+'/freeze.py','post_'+tag],tag+'.post.log');assert post==0
 print(tag,'rc',rc,flush=True)
 if rc!=0:break
heart('capture_finished',KEY)
