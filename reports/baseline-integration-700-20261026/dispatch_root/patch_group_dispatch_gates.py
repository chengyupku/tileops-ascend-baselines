import pathlib,fcntl,ast,shutil,os
D=pathlib.Path(__file__).resolve().parents[1];P=D/'surge_tla_attention97'
gate='''
def ensure_dispatch_enabled():
 state=json.loads((P.parent/'STATE.json').read_text());group=json.loads((P/'GROUP.json').read_text())
 if not state.get('enabled') or state.get('phase') in ('paused','stopped','complete') or group.get('enabled') is False or group.get('phase') in ('paused','stopped','complete','dispatch_exhausted'):
  raise SystemExit('Dispatch disabled by current task/group state; no native spawn')
 limits=[datetime.datetime.fromisoformat(x).timestamp() for x in (state['deadline_utc'],group['deadline_utc'],'2026-10-27T05:31:00+00:00')]
 if time.time()>=min(limits)-60:raise SystemExit('Original deadline dispatch margin reached; no native spawn')

'''
with (P/'CHILD_DISPATCH.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 for name in ['replace_worker.py','spawn_children.py']:
  p=P/name;s=p.read_text()
  if 'def ensure_dispatch_enabled()' in s:continue
  shutil.copy2(p,D/'dispatch_root'/('tla_'+name+'.before_deadline'))
  if name=='replace_worker.py':
   needle="P=pathlib.Path(__file__).resolve().parent\n";assert needle in s;s=s.replace(needle,needle+gate)
   s=s.replace(" fcntl.flock(lock,fcntl.LOCK_EX)\n", " fcntl.flock(lock,fcntl.LOCK_EX)\n ensure_dispatch_enabled()\n")
   s=s.replace(" z=subprocess.run(cmd,capture_output=True,text=True);rec=", " ensure_dispatch_enabled()\n z=subprocess.run(cmd,capture_output=True,text=True);rec=")
  else:
   needle="P=pathlib.Path(__file__).resolve().parent;W=pathlib.Path('/root/dyq/workspace');D=P.parent\n";assert needle in s;s=s.replace(needle,needle+gate)
   s=s.replace(" fcntl.flock(lock,fcntl.LOCK_EX)\n", " fcntl.flock(lock,fcntl.LOCK_EX)\n ensure_dispatch_enabled()\n")
   s=s.replace("  began=time.time();r=subprocess.run(cmd", "  ensure_dispatch_enabled()\n  began=time.time();r=subprocess.run(cmd")
  ast.parse(s);tmp=p.with_name(p.name+'.dispatch.tmp');tmp.write_text(s);os.replace(tmp,p)
 print('Existing TLA initial/continuation launchers now recheck enabled state and fixed deadline before every native spawn; lock and NPU guard behavior unchanged')
