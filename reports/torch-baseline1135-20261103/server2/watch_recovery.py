import pathlib,json,time,subprocess,datetime
P=pathlib.Path(__file__).resolve().parent
code="""import pathlib,json,time,shutil
p=pathlib.Path('/opt/torch-baseline1135-20261103/server2');s=json.loads((p/'STATE.json').read_text());r=json.loads((p/'RETRY.json').read_text());pending={k:v for k,v in s.items() if v['status'] not in ('completed','accepted')};blocked={k:v for k,v in pending.items() if r.get(k,{}).get('at',0)<=v['at']}
errors={}
for k,v in blocked.items():
 root=p/'runs'/k/v['attempt'];errors[k]={'key':v['key'],'errors':[json.loads(f.read_text()).get('error') for f in root.glob('*failure.json')]}
print(json.dumps({'remote_at':time.time(),'pending':len(pending),'blocked':errors,'heartbeat':json.loads((p/'HEARTBEAT.json').read_text()),'shm_free':shutil.disk_usage('/dev/shm').free}))
"""
end=time.time()+1800
while time.time()<end:
 r=subprocess.run(['ssh','-T','-o','ClearAllForwardings=yes','-o','ConnectTimeout=10','openlibing-DevEnv-055409','python3 -'],input=code,text=True,capture_output=True,timeout=30)
 d={'at':time.time(),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'ssh_rc':r.returncode}
 if r.returncode==0:d.update(json.loads(r.stdout))
 else:d['error']=r.stderr
 count=0
 for f in (P/'runs').glob('*/a*/ARCHIVE.json'):
  if json.loads(f.read_text()).get('accepted'):count+=1
 d['accepted']=count
 with (P/'RECOVERY_WATCH.jsonl').open('a') as f:f.write(json.dumps(d)+'\n')
 print(json.dumps(d),flush=True)
 time.sleep(30)
