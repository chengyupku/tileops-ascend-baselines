"""Read-only worker/process/command snapshot; no imports of device libraries."""
import sys,json,pathlib,sqlite3,time,datetime,collections,os
host=sys.argv[1]; selection=json.loads(sys.argv[2]); now=time.time()
base=pathlib.Path('/root/dyq/codex-workers' if host=='local' else '/opt/tileops-all-pairs/control/codex-workers')
db=sqlite3.connect('file:'+str(base/'workers.sqlite')+'?mode=ro',uri=True)
def read(p,default=None):
 try:return json.loads(pathlib.Path(p).read_text())
 except Exception:return default
def epoch(v):
 if isinstance(v,(int,float)):return v
 try:return datetime.datetime.fromisoformat(v.replace('Z','+00:00')).timestamp()
 except Exception:return None
def alive(pid,expected):
 try:
  a=pathlib.Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()
  return a[0]!='Z' and pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()+':'+a[19]==expected
 except Exception:return False
jobs=[]
for ident,status,raw in db.execute('select id,status,data from jobs'):
 d=json.loads(raw)
 if ident not in selection['roots']+selection['workers'] and not (d.get('created_at',0)>now-6*3600 and any('full_rollout_20261024' in p for p in d.get('write',[]))):continue
 r={k:d.get(k) for k in ('id','parent','status','created_at','finished_at','ended_at','exit_code','timeout','model','effort','write','child_pid','supervisor_pid','directory','home')}
 r['alive']=alive(d.get('child_pid'),d.get('child_stamp'));r['supervisor_alive']=alive(d.get('supervisor_pid'),d.get('supervisor_stamp'))
 r['role']='root' if ident in selection['roots'] else 'worker';r['selected_active']=ident in selection['workers'];r['age_s']=now-d['created_at'];directory=pathlib.Path(d['directory']);r['events_mtime']=None
 commands=[];messages=[];errors=[];thread=None
 ep=directory/'events.jsonl'
 if ep.exists():
  r['events_mtime']=ep.stat().st_mtime
  for line in ep.open():
   try:e=json.loads(line)
   except Exception:continue
   if e.get('type')=='thread.started':thread=e.get('thread_id')
   item=e.get('item',{})
   if e.get('type')=='item.completed' and item.get('type')=='command_execution':commands.append({'command':item.get('command'),'exit_code':item.get('exit_code'),'output_tail':item.get('aggregated_output','')[-1800:]})
   if item.get('type')=='agent_message':messages.append(item.get('text','')[:1600])
   if e.get('type') in ('error','turn.failed'):errors.append(e)
 r.update(thread=thread,command_count=len(commands),command_nonzero=sum(c.get('exit_code') not in (None,0) for c in commands),commands=commands if ident in selection['workers'] else commands[-5:],messages=messages[-5:],errors=errors[-5:])
 r['stderr_tail']=(directory/'stderr.log').read_text(errors='replace')[-1800:] if (directory/'stderr.log').exists() else ''
 r['result_tail']=(directory/'result.txt').read_text(errors='replace')[-2500:] if (directory/'result.txt').exists() else ''
 if ident in selection['workers']:
  logs=list(pathlib.Path(d['home']['path']).glob('sessions/2026/10/*/*'+str(thread)+'.jsonl'))
  calls=[]; pending={};r['rollouts']=[str(p) for p in logs]
  for p in logs:
   for line in p.open():
    try:e=json.loads(line)
    except Exception:continue
    q=e.get('payload',{});t=epoch(e.get('timestamp'))
    if e.get('type')!='response_item' or not t:continue
    if q.get('type') in ('function_call','custom_tool_call'):
     c={'start':t,'name':q.get('name'),'input':q.get('arguments',q.get('input',''))};pending[q.get('call_id')]=c;calls.append(c)
    if q.get('type') in ('function_call_output','custom_tool_call_output') and q.get('call_id') in pending:
     c=pending.pop(q['call_id']);c['end']=t;c['duration_s']=max(0,t-c['start'])
     out=q.get('output','');c['output_tail']=str(out)[-800:]
  r['calls']=calls
  r['artifacts']=[]
  for wd in d.get('write',[]):
   root=pathlib.Path(wd)
   if not root.is_dir():continue
   for current,dirs,files in os.walk(root):
    dirs[:]=[v for v in dirs if v not in ('source','sources','build','vendor','__pycache__','.git')]
    for f in files:
     if not f.endswith(('.json','.jsonl','.log')):continue
     p=pathlib.Path(current)/f
     if p.stat().st_size>2000000:continue
     if f.endswith('.json') and ('guard' in f.lower() or 'checker' in f.lower() or f in ('RESULT.json','SCOPE.json','TIMING.json')):
      v=read(p)
      if v is not None:r['artifacts'].append({'path':str(p),'mtime':p.stat().st_mtime,'data':v})
     elif 'guard' in f.lower() and f.endswith(('.log','.jsonl')):
      r['artifacts'].append({'path':str(p),'mtime':p.stat().st_mtime,'text':p.read_text(errors='replace')[-100000:]})
 jobs.append(r)
queue=[]
for p in pathlib.Path('/run/lock/ascend-npu/device-0.queue').glob('*'):
 if p.is_file():queue.append({'path':str(p),'data':read(p),'mtime':p.stat().st_mtime})
print(json.dumps({'host':host,'host_now':now,'jobs':jobs,'guard_lock_metadata':read('/run/lock/ascend-npu/device-0.lock'),'queue':queue},ensure_ascii=False))
