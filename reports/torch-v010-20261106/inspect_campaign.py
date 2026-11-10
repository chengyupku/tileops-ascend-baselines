"""Read-only live inspection, preserving timestamped process/guard/ledger evidence."""
import pathlib,json,subprocess,shlex,datetime,concurrent.futures
P=pathlib.Path(__file__).resolve().parent;hosts=json.load(open(P/'ASSIGNMENT.json'))['hosts'];stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');out=P/'inspections'/stamp;out.mkdir(parents=True)
code=r'''
import pathlib,json,os,subprocess,time,shutil
P=pathlib.Path('/opt/torch-v010-20261106');H=P/HOST
def read(p,default=None):
 try:return json.loads(p.read_text())
 except FileNotFoundError:return {} if default is None else default
hb=read(H/'HEARTBEAT.json');state=read(H/'STATE.json');pid=hb.get('pid');proc=pathlib.Path('/proc')/str(pid)
try:actual_ticks=proc.joinpath('stat').read_text().split()[21]
except FileNotFoundError:actual_ticks=None
processes=[]
for d in pathlib.Path('/proc').iterdir():
 if not d.name.isdigit():continue
 try:
  cmd=d.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  if str(P) not in cmd or 'python -c' in cmd:continue
  st=d.joinpath('stat').read_text().split();processes.append({'pid':int(d.name),'start_ticks':st[21],'cpu_ticks':int(st[13])+int(st[14]),'rss_pages':int(st[23]),'command':cmd})
 except (OSError,IndexError,ValueError):pass
env=dict(os.environ);env['LD_LIBRARY_PATH']='/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64'
g=subprocess.run(['/opt/tileops-ci/runtime/bin/python','-I','-S',str(P/'tools/npu_guard.py'),'status','--json'],capture_output=True,text=True,env=env,timeout=45)
try:guard=json.loads(g.stdout)
except ValueError:guard={'stdout':g.stdout,'stderr':g.stderr,'returncode':g.returncode}
print(json.dumps({'checked_at':time.time(),'heartbeat':hb,'controller_identity_matches':actual_ticks==str(hb.get('start_ticks')),'actual_start_ticks':actual_ticks,'state':state,'processes':processes,'guard':guard,'disk_free_GiB':shutil.disk_usage(P).free/2**30,'shm_free_GiB':shutil.disk_usage('/dev/shm').free/2**30,'pipeline_tail':(H/'pipeline.log').read_text(errors='replace')[-12000:] if (H/'pipeline.log').exists() else ''}))
'''
def inspect(h):
 cmd=['ssh','-T','-o','BatchMode=yes','-o','ClearAllForwardings=yes','-o','ConnectTimeout=15',hosts[h],shlex.join(['/opt/tileops-ci/runtime/bin/python','-I','-S','-c','HOST='+repr(h)+'\n'+code])]
 r=subprocess.run(cmd,capture_output=True,text=True,check=True,timeout=60);data=json.loads(r.stdout);(out/(h+'.json')).write_text(json.dumps(data,indent=2))
 progress=json.load(open(P/h/'PROGRESS.json'));accepted={k for k,v in progress.items() if v.get('accepted')};done={k for k,v in data['state'].items() if v.get('status') in ('completed','accepted')};assigned={r['id'] for r in json.load(open(P/h/'PLAN.json'))['rows']};pending={k:v for k,v in data['state'].items() if k in assigned and v.get('status') not in ('completed','accepted')}
 return h,{'controller_identity_matches':data['controller_identity_matches'],'accepted':len(accepted),'remote_done':len(done),'archive_backlog':sorted(done-accepted),'local_only':sorted(accepted-done),'pending':pending,'disk_free_GiB':data['disk_free_GiB'],'shm_free_GiB':data['shm_free_GiB'],'heartbeat':data['heartbeat']}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:summary=dict(pool.map(inspect,hosts))
(out/'SUMMARY.json').write_text(json.dumps(summary,indent=2));(P/'LATEST_INSPECTION.json').write_text(json.dumps({'path':str(out),'summary':summary},indent=2));print(json.dumps(summary,indent=2))
