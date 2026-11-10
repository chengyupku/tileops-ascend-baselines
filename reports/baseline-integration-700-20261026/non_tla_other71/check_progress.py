import json,datetime,sqlite3
from pathlib import Path
D=Path('/root/dyq/workspace/reports/baseline-integration-700-20261026');P=D/'non_tla_other71';db=sqlite3.connect('file:/root/dyq/codex-workers/workers.sqlite?mode=ro',uri=True)
def read(p,d=None):
 try:return json.loads(p.read_text())
 except:return d
scope={tuple(r['key']) for r in read(P/'SCOPE.json')};accepted={tuple(r['key']) for r in read(D/'ACCEPTED.json')['rows']};jobs=read(P/'CHILD_JOBS.json',{'jobs':[]})['jobs'];children=[]
for j in jobs:
 row=db.execute('select status,data from jobs where id=?',(j['native']['id'],)).fetchone();data=json.loads(row[1]);pid=data.get('child_pid');stamp=data.get('child_stamp');live=False
 try:
  st=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split();live=st[0]!='Z' and Path('/proc/sys/kernel/random/boot_id').read_text().strip()+':'+st[19]==stamp
 except:pass
 q=Path(j['directory']);children.append({'name':q.name,'id':j['native']['id'],'status':row[0],'identity_live':live,'heartbeat':read(q/'HEARTBEAT.json'),'result_exists':(q/'RESULT.json').exists()})
h={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'accepted_in_scope':len(scope&accepted),'accepted_delta':len(scope&accepted),'ready':len(read(P/'READY_QUEUE.json')['tasks'])-len(jobs),'running':sum(c['identity_live'] for c in children),'device_queue':0,'next_action':'review CPU original-interface findings; no NPU authorized','children':children}
(P/'HEARTBEAT.json').write_text(json.dumps(h,ensure_ascii=False,indent=2)+'\n');print(json.dumps(h,ensure_ascii=False))
