"""Stop only this campaign controller immediately after a finished device receipt."""
import pathlib,json,time,os,signal,sys
P=pathlib.Path(__file__).resolve().parent;H=P/sys.argv[1]
h=json.loads((H/'HEARTBEAT.json').read_text());pid=h['pid'];ticks=h['start_ticks'];(H/'STOP').touch()
def alive():
 try:return pathlib.Path(f'/proc/{pid}/stat').read_text().split()[21]==ticks
 except FileNotFoundError:return False
def ended():
 p=H/'pipeline'/h.get('batch','unknown')/'events.jsonl'
 if not p.exists():return []
 es=[]
 for line in p.read_text().splitlines():
  try:e=json.loads(line)
  except json.JSONDecodeError:continue
  if e.get('event')=='phase_end' and e.get('phase')=='device':es.append(e)
 return es
before=len(ended())
while alive():
 if len(ended())>before or h.get('status') in ('waiting_disk','waiting_repairs'):
  os.kill(pid,signal.SIGTERM);break
 time.sleep(.1)
while alive():time.sleep(.2)
(H/'DRAIN_BOUNDARY.json').write_text(json.dumps({'at':time.time(),'pid':pid,'initial_completed':before,'finished_events':ended()},indent=2))
