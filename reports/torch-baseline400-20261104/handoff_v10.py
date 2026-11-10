"""Disk admission/closed-cancellation repair; measured worker remains frozen v9."""
import pathlib,json,time,os,sys
p=pathlib.Path(__file__).resolve().parent;h=p/'server1'
try:assert pathlib.Path('/proc/2697795/stat').read_text().split()[21]!='278348070'
except FileNotFoundError:pass
state=json.loads((h/'STATE.json').read_text());retry=json.loads((h/'RETRY.json').read_text());changed=[]
for d in sorted((h/'runs').glob('*/a*'),key=lambda d:int(d.name[1:])):
 if not (d/'CANCELLED.json').exists():continue
 tid=d.parent.name
 if state.get(tid,{}).get('status') in ('completed','accepted','failed','invalid_guard'):continue
 state[tid]={'status':'cancelled','at':time.time(),'attempt':d.name,'key':json.loads((d/'task.json').read_text())['key'],'error':'Historical cancelled attempt retained; v10 disk admission and archival repair'}
 retry[tid]={'at':time.time()+.001,'reason':'v10 archives closed cancelled raw and waits for 58GiB free before 64k KDA; 51.5GiB other jobs; final receipt SHM spool enabled. Original v9 worker/checker unchanged.'}
 changed.append(tid)
for name,data in [('STATE',state),('RETRY',retry)]:
 t=h/(name+'.v10.tmp');t.write_text(json.dumps(data,indent=2));t.replace(h/(name+'.json'))
(h/'V10_HANDOFF.json').write_text(json.dumps({'at':time.time(),'retries':sorted(set(changed)),'old_pid_exited':True,'worker':'worker_server1_v9.py','scheduler':'supervisor_server1_v10.py'},indent=2))
(h/'STOP').unlink(missing_ok=True)
os.execv(sys.executable,[sys.executable,str(p/'supervisor_server1_v10.py'),'server1'])
