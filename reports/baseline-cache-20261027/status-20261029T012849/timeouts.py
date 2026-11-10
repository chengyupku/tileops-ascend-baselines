import json,time,sys,collections
from pathlib import Path
h=sys.argv[1];r=Path('reports/baseline-cache-20261027');start=1793217600-(0 if h=='local' else 93.08)
rows=[json.loads(x) for x in (r/h/'journal.jsonl').read_text().splitlines()]
timeouts=[x for x in rows if x.get('at',0)>=start and x.get('exit_code')==124]
res=[]
for x in timeouts:
 p=Path(x['guard']);g=json.loads(p.read_text()) if p.exists() else {}
 res.append({'id':x['id'],'key':x['key'],'seconds':x['wall_seconds'],'guard_summary':{k:v for k,v in g.items() if k in ['status','reason','exit_code','timed_out','timeout','performance_valid','child_returncode','conflicts']}})
out={'host':h,'timeouts':res,'timeout_wall_minutes':sum(x['wall_seconds'] for x in timeouts)/60}
Path('reports/baseline-cache-20261027/status-20261029T012849/'+h+'-timeouts.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(out,ensure_ascii=False))

