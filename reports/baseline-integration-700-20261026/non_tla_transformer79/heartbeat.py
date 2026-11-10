import json,pathlib,sqlite3,datetime
G=pathlib.Path(__file__).resolve().parent;D=G.parent
def read(p,default=None):
 try:return json.loads(p.read_text())
 except (OSError,ValueError):return default

def live(v):
 try:
  s=pathlib.Path('/proc/'+str(v['child_pid'])+'/stat').read_text().rsplit(')',1)[1].split();return s[0]!='Z' and pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()+':'+s[19]==v['child_stamp']
 except (OSError,KeyError):return False

db=sqlite3.connect('file:/root/dyq/codex-workers/workers.sqlite?mode=ro',uri=True);rows=[]
for j in read(G/'CHILD_JOBS.json',{'jobs':[]})['jobs']:
 r=db.execute('select status,data from jobs where id=?',(j['native']['id'],)).fetchone();v=json.loads(r[1]);p=pathlib.Path(j['directory']);rows.append({'name':p.name,'id':v['id'],'status':r[0],'identity_live':live(v),'parent':v['parent'],'notification':v.get('notification'),'heartbeat':read(p/'HEARTBEAT.json'),'result_exists':(p/'RESULT.json').exists()})
keys={tuple(x['key']) for x in read(G/'SCOPE.json',[])};acc={tuple(x['key']) for x in read(D/'ACCEPTED.json',{'rows':[]})['rows']};h={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':79,'accepted_delta':len(keys&acc),'ready':max(0,len(read(G/'READY_QUEUE.json',{'tasks':[]})['tasks'])-len(rows)),'running':sum(r['status'] in ['running','starting','stopping'] or r['identity_live'] for r in rows),'npu_queue':'unknown for active device workers; see individual guard receipts','children':rows,'next_action':'Review new original API and deployment evidence; only changed-condition legal device routes may enter next queue'}
(G/'HEARTBEAT.json').write_text(json.dumps(h,ensure_ascii=False,indent=2)+'\n');print(json.dumps({**{k:v for k,v in h.items() if k!='children'},'children':[{'name':r['name'],'status':r['status'],'identity_live':r['identity_live'],'stage':(r.get('heartbeat') or {}).get('phase',(r.get('heartbeat') or {}).get('stage'))} for r in rows if r['status'] in ['running','starting','stopping'] or r['identity_live']]},ensure_ascii=False))

# Controller feedback must be visible to this native exec worker; queue delivery alone is insufficient.
feedback = G / 'CONTROLLER_ACTION_20261027T0705.txt'
if feedback.exists() and not (G / 'ACTION_ACK.json').exists():
 print('CONTROLLER_ACTION_PENDING: ' + feedback.read_text()[:2600])
