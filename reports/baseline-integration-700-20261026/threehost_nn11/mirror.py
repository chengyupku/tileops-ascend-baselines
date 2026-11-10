from pathlib import Path
import json,time,subprocess,datetime,sys
P=Path(__file__).resolve().parent;host,attempt=sys.argv[1:];R='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_nn11';ssh='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
while True:
 r=subprocess.run([ssh,host,'cat '+R+'/HEARTBEAT.json; test -f '+R+'/'+attempt+'/GUARD_RESULT.json'],capture_output=True,text=True)
 try:d=json.loads(r.stdout)
 except ValueError:d={'remote_output':r.stdout,'remote_error':r.stderr}
 d['mirror_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();d['lastprogress']=d['mirror_at'];d['transport_returncode']=r.returncode
 (P/'HEARTBEAT.json').write_text(json.dumps(d,indent=2))
 if r.returncode==0:break
 time.sleep(25)
print('remote guard finished')
