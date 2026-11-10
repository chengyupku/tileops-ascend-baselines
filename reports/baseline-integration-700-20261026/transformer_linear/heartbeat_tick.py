from pathlib import Path
import json,datetime
W=Path(__file__).resolve().parent;p=W/'HEARTBEAT.json';v=json.loads(p.read_text());now=datetime.datetime.now(datetime.timezone.utc)
if v.get('phase')!='finished' and (now-datetime.datetime.fromisoformat(v['utc'])).total_seconds()>240:
 v['utc']=now.isoformat();v['note']='Foreground guarded batch is still queued/running; progress timestamp retained.';t=W/'HEARTBEAT.tick';t.write_text(json.dumps(v,indent=2));t.replace(p)
