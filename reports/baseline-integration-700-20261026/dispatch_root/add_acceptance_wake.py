import pathlib,fcntl,ast
D=pathlib.Path(__file__).resolve().parents[1];p=D/'refill_fleet.py'
helper='''def wake_acceptance_for_new_publish(state,db,record):
 """Event-driven authorized receiver restart; never re-review an unchanged queue."""
 queue=read(D/'ROOT_CONTROL_QUEUE.json',{'tasks':[]})
 entry=next((e for e in queue['tasks'] if e.get('name')=='acceptance_root'),None)
 if not entry or entry.get('ready') or not entry.get('job_id'):return
 row=db.execute('select status,data from jobs where id=?',(entry['job_id'],)).fetchone()
 if not row:return
 dat=json.loads(row[1])
 if row[0] in ACTIVE or live(dat.get('child_pid'),dat.get('child_stamp')):return
 accepted={tuple(r['key']) for r in read(D/'ACCEPTED.json',{'rows':[]})['rows']}
 candidates=[]
 for name in state.get('coordinator_groups',[]):
  path=D/name/'PUBLISH_QUEUE.json'
  for item in read(path,{'entries':[]}).get('entries',[]):
   k=key(item)
   if not k or k in accepted or not (item.get('ready_to_publish') or item.get('status')=='verified_ready_for_parent_serial_publication'):continue
   receipt_times={}
   for field in ('review','checker','native','capture','execution','registration_pins'):
    if isinstance(item.get(field),str):
     f=pathlib.Path(item[field]);f=f if f.is_absolute() else D/name/f
     try:receipt_times[field]=f.stat().st_mtime_ns
     except OSError:receipt_times[field]=None
   semantic={k:v for k,v in item.items() if k not in ('utc','at','updated_at','heartbeat')}
   candidates.append({'group':name,'entry':semantic,'receipt_mtimes':receipt_times})
 if not candidates:return
 signature=hashlib.sha256(json.dumps(candidates,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
 if entry.get('last_publish_wake_signature')==signature:return
 old=entry['job_id'];entry.setdefault('prior_job_ids',[]).append(old);entry.pop('job_id');entry['ready']=True;entry['last_publish_wake_signature']=signature
 entry['changed_condition']='New unaccepted coordinator-approved PUBLISH_QUEUE evidence since terminal acceptance receiver; existing authority unchanged'
 parent=read(D/'dispatch_root/SELF_JOB.json')
 if parent:
  r=db.execute('select status,data from jobs where id=?',(parent['id'],)).fetchone()
  if r and r[0] in {'starting','running'}:entry['parent_self_job']=str(D/'dispatch_root/SELF_JOB.json')
  else:entry.pop('parent_self_job',None)
 put(D/'ROOT_CONTROL_QUEUE.json',queue)
 put(D/'acceptance_root/WAKE_REQUEST.json',{'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'previous_job':old,'signature':signature,'candidates':candidates,'reason':entry['changed_condition']})
 record['rejected'].append({'control':'acceptance_root','reason':'new publication evidence queued existing authorized receiver'})

'''
with (D/'FLEET_REFILL.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);s=p.read_text();assert 'def wake_acceptance_for_new_publish' not in s;s=s.replace('import datetime,fcntl,json,os,pathlib,sqlite3,subprocess,time','import datetime,fcntl,json,os,pathlib,sqlite3,subprocess,time,hashlib');s=s.replace('def main():',helper+'def main():');s=s.replace('  dispatch_controls(state,db,deadline,record)','  wake_acceptance_for_new_publish(state,db,record)\n  dispatch_controls(state,db,deadline,record)');ast.parse(s);p.write_text(s)
print('Added event-driven acceptance successor for genuinely new unaccepted publish entries; unchanged evidence cannot trigger repeated root audits')
