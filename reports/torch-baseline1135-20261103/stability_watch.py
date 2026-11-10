"""CPU-only live samples. No device imports or actions."""
import pathlib,json,time,subprocess,shlex,concurrent.futures
P=pathlib.Path(__file__).resolve().parent;HOSTS=json.loads((P/'ASSIGNMENT.json').read_text())['hosts'];R='/opt/torch-baseline1135-20261103'
def sample(h):
 code='''import pathlib,json,time
p=pathlib.Path(ROOT)/HOST;h=json.loads((p/'HEARTBEAT.json').read_text());ev=p/'pipeline'/h.get('batch','x')/'events.jsonl';active={};events=[]
if ev.exists():
 for line in ev.read_text().splitlines():
  try:e=json.loads(line)
  except json.JSONDecodeError:continue
  events.append(e)
  if e['event']=='phase_start':active[(e['id'],e['phase'])]=e
  if e['event']=='phase_end':active.pop((e['id'],e['phase']),None)
print(json.dumps({'remote_now':time.time(),'heartbeat':h,'active':list(active.values()),'events':events}))
'''.replace('ROOT',repr(R)).replace('HOST',repr(h))
 r=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ClearAllForwardings=yes',HOSTS[h],'python3 -c '+shlex.quote(code)],capture_output=True,text=True,timeout=25)
 d=json.loads(r.stdout) if r.returncode==0 else {'error':r.stderr};best={}
 for f in (P/h/'runs').glob('*/a*/ARCHIVE.json'):
  try:a=json.loads(f.read_text())
  except (FileNotFoundError,json.JSONDecodeError):continue
  tid=f.parent.parent.name
  if a.get('accepted') or tid not in best or (not best[tid].get('accepted') and a['at']>best[tid]['at']):best[tid]=a
 d['accepted']=sum(bool(a.get('accepted')) for a in best.values());d['pending']=[k for k,a in best.items() if not a.get('accepted')];return h,d
if __name__=='__main__':
 out=P/'stability';out.mkdir(exist_ok=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  while True:
   try:
    row={'at':time.time(),'hosts':dict(pool.map(sample,HOSTS))}
    with (out/'samples.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    (out/'LATEST.json.tmp').write_text(json.dumps(row,indent=2));(out/'LATEST.json.tmp').replace(out/'LATEST.json')
   except Exception as e:print(type(e).__name__,str(e),flush=True)
   time.sleep(30)
