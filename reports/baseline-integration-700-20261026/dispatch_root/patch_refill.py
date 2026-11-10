import pathlib,fcntl,shutil,ast
D=pathlib.Path(__file__).resolve().parents[1];p=D/'refill_fleet.py'
with (D/'FLEET_REFILL.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);s=p.read_text();before=s
 s=s.replace(" if not entry.get('ready') or not entry.get('approved_by_coordinator'):return 'not coordinator approved'", " if entry.get('enabled') is False or entry.get('phase') in ('paused','stopped','complete'):return 'entry disabled'\n if not entry.get('ready') or not entry.get('approved_by_coordinator'):return 'not coordinator approved'")
 s=s.replace("  if not entry.get('ready') or entry.get('job_id'):continue", "  if time.time()>=deadline-60:record['stopped']='deadline dispatch margin';break\n  if entry.get('enabled') is False or entry.get('phase') in ('paused','stopped','complete'):continue\n  if not entry.get('ready') or entry.get('job_id'):continue")
 s=s.replace("  active=0\n  for status,data", "  dispatch_parent=state['parent'];dispatch_home='codex-7'\n  if entry.get('parent_self_job'):\n   ps=read(entry['parent_self_job']);pr=db.execute('select status,data from jobs where id=?',(ps['id'],)).fetchone()\n   if not pr or pr[0] not in ACTIVE:record['rejected'].append({'control':entry['name'],'reason':'requested real parent inactive'});continue\n   dispatch_home=ps['home']['name'];dispatch_parent=None\n   for line in (pathlib.Path(ps['directory'])/'events.jsonl').open():\n    e=json.loads(line)\n    if e.get('type')=='thread.started':dispatch_parent=e['thread_id'];break\n   if not dispatch_parent:record['rejected'].append({'control':entry['name'],'reason':'no real parent thread'});continue\n  active=0\n  for status,data")
 s=s.replace("(state['parent'],)):","(dispatch_parent,)):")
 s=s.replace("'--parent',state['parent'],'--parent-home','codex-7'", "'--parent',dispatch_parent,'--parent-home',dispatch_home")
 s=s.replace("  for p in groups:\n   with", "  for p in groups:\n   if time.time()>=deadline-60:record['stopped']='deadline dispatch margin';break\n   group_control=read(p/'GROUP.json',{})\n   if group_control.get('enabled') is False or group_control.get('phase') in ('paused','stopped','complete'):\n    record['groups'].append({'name':p.name,'dispatch_disabled':True});continue\n   with")
 s=s.replace("selfjob=read(p/'SELF_JOB.json');parentrow", "selfjob=read(group_control.get('supervisor_self_job',p/'SELF_JOB.json'));parentrow")
 s=s.replace("    for entry in queue:\n     if active", "    for entry in queue:\n     if time.time()>=deadline-60:record['stopped']='deadline dispatch margin';break\n     if active")
 ast.parse(s);assert s!=before
 shutil.copy2(p,D/'dispatch_root/refill_fleet.before.py');p.write_text(s)
 print('Patched real-parent control dispatch, explicit group supervision pointer, disabled-entry/group gates and per-launch deadline margin; syntax valid')
