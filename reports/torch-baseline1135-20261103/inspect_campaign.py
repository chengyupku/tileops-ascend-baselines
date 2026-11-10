"""Read-only CPU inspection of both host identities, guard, disk and task status."""
import pathlib,json,subprocess,time,concurrent.futures,shlex,datetime
P=pathlib.Path(__file__).resolve().parent;R='/opt/torch-baseline1135-20261103'
HOSTS=json.loads((P/'ASSIGNMENT.json').read_text())['hosts']
def inspect(h):
 code='''import json,pathlib,os,shutil,subprocess,time
p=pathlib.Path(ROOT)/HOST
def read(f):
 try:return json.loads((p/f).read_text())
 except FileNotFoundError:return {}
h=read('HEARTBEAT.json');pid=h.get('pid');alive=False
if pid:
 try:alive=pathlib.Path('/proc/'+str(pid)+'/stat').read_text().split()[21]==h['start_ticks']
 except FileNotFoundError:pass
env=dict(os.environ,LD_LIBRARY_PATH='/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/cann-9.1.0/lib64')
g=subprocess.run(['python3','-I','-S',ROOT+'/tools/npu_guard.py','status','--json'],env=env,capture_output=True,text=True,timeout=45)
s=read('STATE.json');pending={k:v for k,v in s.items() if v['status'] not in ('completed','accepted')}
print(json.dumps({'remote_time':time.time(),'heartbeat':h,'identity_valid':alive,'free_GiB':shutil.disk_usage('/').free/2**30,'guard_rc':g.returncode,'guard':g.stdout,'guard_error':g.stderr,'pending':pending,'state':s}))
'''.replace('ROOT',repr(R)).replace('HOST',repr(h))
 r=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ClearAllForwardings=yes',HOSTS[h],'python3 -c '+shlex.quote(code)],capture_output=True,text=True,timeout=75)
 return {'host':h,'at':time.time(),'rc':r.returncode,'observation':json.loads(r.stdout) if r.returncode==0 else None,'stderr':r.stderr}
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:result=list(pool.map(inspect,HOSTS))
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');out=P/'inspections'/stamp;out.mkdir(parents=True)
 (out/'OBSERVATIONS.json').write_text(json.dumps(result,indent=2))
 for r in result:
  d=r['observation'] or {};print(r['host'],'identity',d.get('identity_valid'),'disk',d.get('free_GiB'),'pending',len(d.get('pending',{})),'rc',r['rc'])
 print(out)
