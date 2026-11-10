#!/usr/bin/env python3
"""Bounded operational projection only; never opens individual results."""
import json,pathlib,sqlite3,sys,datetime
D=pathlib.Path(__file__).resolve().parent

def read(p,default):
 try:return json.loads(p.read_text())
 except (OSError,ValueError):return default

def identity(j):
 try:
  s=pathlib.Path('/proc/%s/stat'%j['child_pid']).read_text().rsplit(')',1)[1].split()
  return s[0]!='Z' and pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()+':'+s[19]==j['child_stamp']
 except (OSError,KeyError,TypeError,IndexError):return False

name=sys.argv[1] if len(sys.argv)>1 else 'dispatch_root'
if name not in ('dispatch_root','surge_tla_attention97','surge_dense_linear92','surge_attention_moe51','recovery_native5'):raise SystemExit('Unknown role')
x=read(D/'LIVE.json',{});out={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'role':name,'aggregate':{k:x.get(k) for k in ('at','parent_accepted_completed','accepted_last_hour','execution_workers','execution_workers_alive','coordinator_workers')},'contract':str(D/'COORDINATOR_CURRENT_CONTRACT.txt'),'groups':[]}
groups=['surge_tla_attention97','surge_dense_linear92','surge_attention_moe51','recovery_native5'] if name=='dispatch_root' else [name]
with sqlite3.connect('file:/root/dyq/codex-workers/workers.sqlite?mode=ro',uri=True) as db:
 for group in groups:
  p=D/group;g=read(p/'GROUP.json',{});a={'name':group,'enabled':g.get('enabled',True),'phase':g.get('phase'),'supervisor_self_job':g.get('supervisor_self_job'),'ready':sum(bool(e.get('ready')) and bool(e.get('approved_by_coordinator')) for e in read(p/'READY_QUEUE.json',{}).get('tasks',[])),'publish_entries':len(read(p/'PUBLISH_QUEUE.json',{}).get('entries',[])),'children':[]}
  for e in read(p/'CHILD_JOBS.json',{}).get('jobs',[]):
   jid=e.get('native',{}).get('id');row=db.execute('select data from jobs where id=?',(jid,)).fetchone()
   if row:
    j=json.loads(row[0]);a['children'].append({'job_id':jid,'status':j.get('status'),'alive':identity(j),'directory':e.get('directory')})
  h=read(p/'DISPATCH_HANDOFF.json',{});a['handoff']={k:h.get(k) for k in ('utc','never_submitted_candidates_total','ready_for_external_dispatch')};out['groups'].append(a)
text=json.dumps(out,ensure_ascii=False,separators=(',',':'))
if len(text.encode())>8192:raise SystemExit('Projection exceeds 8KiB; narrow role')
print(text)
