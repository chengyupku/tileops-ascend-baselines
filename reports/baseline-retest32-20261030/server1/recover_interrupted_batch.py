"""CPU-only recovery of complete receipts after supervisor loss; preserve incomplete attempt."""
import json,sys,time,fcntl,hashlib,os,shutil
from pathlib import Path
r=Path('/opt/tileops-baseline-cache/workspace');sys.path.insert(0,str(r/'tools'))
from baseline_cache_guard_binding import bind_guard,capture_guard
h=r/'reports/baseline-retest32-20261030/server1';lock=(h/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
a=json.loads((h/'ACTIVE.json').read_text());assert not Path('/proc/'+str(a['identity']['pid'])).exists()
def append(p,x):
 with p.open('a') as f:f.write(json.dumps(x)+'\n');f.flush();os.fsync(f.fileno())
def put(p,x):
 q=p.with_suffix('.recoverytmp');q.write_text(json.dumps(x,indent=2));q.replace(p)
latest={x['id']:x for x in map(json.loads,(h/'journal.jsonl').read_text().splitlines())};recovered=[];incomplete=[]
for j in a['jobs']:
 if latest.get(j['id'],{}).get('status')=='cached':continue
 o=Path(j['out']);p=o/'measurement.json';gp=Path(a['batch'])/j['id']/'guard.json'
 if not p.exists() or not gp.exists():incomplete.append({'id':j['id'],'out':str(o),'reason':'Interrupted before complete measurement/guard;preserve evidence and rerun only after host recovery'});continue
 m=json.loads(p.read_text());cpath=Path(m['correctness_receipt']);c=json.loads(cpath.read_text());assert m['status']=='measured_pending_guard';assert c['status']=='verified' and c['inputs_preserved'] and not c.get('provider_source_changed') and not c.get('oracle_source_changed');assert hashlib.sha256(cpath.read_bytes()).hexdigest()==m['correctness_sha256'];assert len(m['end_to_end_samples_ms'])==15;assert m['identity']['environment']['aicore_count']==32
 m.update(status='cached',performance_valid=True,guard_receipt=str(gp));m['guard_binding']=bind_guard(m,json.loads((o/'prepare.json').read_text())['guard_before'],capture_guard(),retrospective=True)
 backup=o/'measurement.before_recovery.json';assert not backup.exists();shutil.copy2(p,backup);put(p,m);append(h/'CACHE.jsonl',m);append(h/'journal.jsonl',{'id':j['id'],'key':j['key'],'at':time.time(),'status':'cached','measurement':str(p),'guard':str(gp),'error':None,'batch':a['batch'],'revision':'interrupted_supervisor_receipt_recovery','recovery':'Complete originalchecker+15samples+validguard;retrospective unchangedguardfile binding;no rerun'});recovered.append(j['id'])
record={'at':time.time(),'recovered':recovered,'incomplete':incomplete,'original_active':a};put(h/'INTERRUPTED_BATCH_RECOVERY.json',record);print(json.dumps({k:v for k,v in record.items() if k!='original_active'}))
