"""Read-only live inspection and local receipt ledger; no device imports."""
import pathlib,json,subprocess,shlex,datetime,collections
P=pathlib.Path(__file__).resolve().parent
at=datetime.datetime.now(datetime.timezone.utc);d=P/'server1/inspections'/at.strftime('%Y%m%dT%H%M%SZ');d.mkdir(parents=True)
code='''import pathlib,json,shutil,collections,time
p=pathlib.Path('/opt/torch-baseline400-20261104/server1');h=json.loads((p/'HEARTBEAT.json').read_text());proc=pathlib.Path('/proc')/str(h['pid']);ticks=proc.joinpath('stat').read_text().split()[21];assert ticks==str(h['start_ticks'])
s=json.loads((p/'STATE.json').read_text());pending=[]
for k,v in s.items():
 if v['status']!='completed':
  out=p/'runs'/k/v['attempt'];pending.append(dict(id=k,state=v,failures={f.name:f.read_text() for f in out.glob('*failure.json')}))
cancelled=[{'attempt':str(f.parent.relative_to(p/'runs')),'done':(f.parent/'DONE.json').exists(),'archived':(f.parent/'ARCHIVED.json').exists()} for f in (p/'runs').glob('*/a*/CANCELLED.json')]
batch=p/'pipeline'/h.get('batch','');eventfile=batch/'events.jsonl';events=eventfile.read_text().splitlines()[-10:] if eventfile.is_file() else []
print(json.dumps({'checked_at':time.time(),'cancelled_attempts':cancelled,'heartbeat':h,'pid_start_verified':True,'command':proc.joinpath('cmdline').read_bytes().replace(bytes([0]),b' ').decode(),'counts':dict(collections.Counter(x['status'] for x in s.values())),'pending':pending,'disk_free_GiB':shutil.disk_usage(p).free/2**30,'shm_free_GiB':shutil.disk_usage('/dev/shm').free/2**30,'latest_events':events,'blocked':json.loads((p/'BLOCKED.json').read_text())}))'''
def remote(cmd):return subprocess.run(['ssh','-T','-o','ClearAllForwardings=yes','openlibing-DevEnv-661495',cmd],text=True,capture_output=True,timeout=60)
r=remote(shlex.join(['python3','-c',code]));assert r.returncode==0,r.stderr;live=json.loads(r.stdout);(d/'REMOTE_CHECK.json').write_text(json.dumps(live,indent=2))
g=remote('export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/cann-9.1.0/lib64; python3 /opt/torch-baseline400-20261104/tools/npu_guard.py status --json');guard=json.loads(g.stdout);(d/'GUARD_STATUS.json').write_text(json.dumps(guard,indent=2))
progress=json.loads((P/'server1/PROGRESS.json').read_text());rows=[]
for tid,v in progress.items():
 if v.get('accepted'):
  archive=pathlib.Path(v['archive']);assert archive.exists() and archive.stat().st_size==v['bytes'];assert v['verified'] and v['measurement']['raw_windows']==15
  rows.append({'id':tid,'sha256':v['sha256'],'bytes':v['bytes']})
(d/'LOCAL_LEDGER.json').write_text(json.dumps({'checked_at':at.isoformat(),'accepted':len(rows),'rows':rows},indent=2))
print(json.dumps({'inspection':str(d),'accepted':len(rows),'remote':live,'guard':guard},ensure_ascii=False))
