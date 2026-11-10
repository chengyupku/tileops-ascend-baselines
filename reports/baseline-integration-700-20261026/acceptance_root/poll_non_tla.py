"""Read-only worker/queue observation; own heartbeat is the only mutation."""
import datetime,json,pathlib,subprocess
R=pathlib.Path('/root/dyq/workspace/reports/baseline-integration-700-20261026');A=R/'acceptance_root'
def read(p):return json.loads(p.read_text())
prev=read(A/'MONITOR.json') if (A/'MONITOR.json').exists() else {};states=prev.get('states',{});active=[];pending=[]
accepted={tuple(x['key']) for x in read(R/'ACCEPTED.json')['rows']}
for group in ['non_tla_transformer79','non_tla_other71']:
 for e in read(R/group/'PUBLISH_QUEUE.json')['entries']:
  if tuple(e['key']) not in accepted:pending.append({'group':group,'key':e['key']})
 for j in read(R/group/'CHILD_JOBS.json')['jobs']:
  n=j['native'];id=n['id'];status=states.get(id,{}).get('status')
  if status not in ['completed','failed','cancelled','lost','timed_out']:
   p=subprocess.run(['codex-worker','status',id],capture_output=True,text=True)
   d=json.loads(p.stdout) if p.returncode==0 else {'status':'unknown'};status=d['status']
  states[id]={'name':j['name'],'status':status}
  if status not in ['completed','failed','cancelled','lost','timed_out']:active.append({'id':id,**states[id]})
x={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'owner':'2c16f31fa2744933','actual_accepted_added':7,'pending':len(pending),'pending_entries':pending,'active':active,'states':states,'blockers':'Await active group result submissions' if active else None}
for name in ['MONITOR.json','HEARTBEAT.json']:(A/name).write_text(json.dumps(x,indent=2)+'\n')
print(json.dumps({k:v for k,v in x.items() if k!='states'},ensure_ascii=False))
