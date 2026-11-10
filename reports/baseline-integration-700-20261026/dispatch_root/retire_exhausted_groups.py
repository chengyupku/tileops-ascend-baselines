import pathlib,json,sqlite3,fcntl,datetime,runpy,os
D=pathlib.Path(__file__).resolve().parents[1];m=runpy.run_path(str(D/'refill_fleet.py'));db=sqlite3.connect('file:/root/dyq/codex-workers/workers.sqlite?mode=ro',uri=True);actions=[]
with (D/'FLEET_REFILL.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 for name in ['surge_dense_linear92','surge_attention_moe51']:
  p=D/name;s=json.loads((p/'SELF_JOB.json').read_text());r=db.execute('select status,data from jobs where id=?',(s['id'],)).fetchone();d=json.loads(r[1]);assert r[0]=='completed' and not m['live'](d.get('child_pid'),d.get('child_stamp'))
  with (p/'CHILD_DISPATCH.lock').open('a') as child:
   fcntl.flock(child,fcntl.LOCK_EX)
   for j in json.loads((p/'CHILD_JOBS.json').read_text())['jobs']:
    r=db.execute('select status,data from jobs where id=?',(j['native']['id'],)).fetchone();d=json.loads(r[1]);assert r[0] not in m['ACTIVE'] and not m['live'](d.get('child_pid'),d.get('child_stamp'))
   assert not any(e.get('ready') for e in json.loads((p/'READY_QUEUE.json').read_text())['tasks'])
   result=json.loads((p/'RESULT.json').read_text());assert len(result['rows'])==len(json.loads((p/'SCOPE.json').read_text()))
   group=json.loads((p/'GROUP.json').read_text());group.update(enabled=False,phase='dispatch_exhausted',dispatch_stop_reason='Coordinator and all children terminal; exact scope delivery complete; no ready executable work or changed-condition replacement. Numerical/API failures are not accepted. Preserve pending publication for acceptance root.',restart_requires='Concrete changed condition or newly released executable scope; explicit supervisor assignment before reenable')
   temp=p/'GROUP.json.dispatch.tmp';temp.write_text(json.dumps(group,ensure_ascii=False,indent=2)+'\n');os.replace(temp,p/'GROUP.json');actions.append({'group':name,'action':'disable exhausted dispatch, preserve full original result and ownership evidence','coordinator_completed':True,'all_children_terminal':True,'scope_rows':len(result['rows'])})
(D/'dispatch_root/EXHAUSTED_GROUPS.json').write_text(json.dumps({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'actions':actions},ensure_ascii=False,indent=2)+'\n');print(json.dumps(actions))
