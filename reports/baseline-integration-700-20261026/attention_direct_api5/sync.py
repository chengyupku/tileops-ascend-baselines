import json,base64
from remote import run,R
p=run("from pathlib import Path\nimport json,base64\nr=Path("+repr(str(R))+")\nprint(json.dumps({str(p.relative_to(r)):base64.b64encode(p.read_bytes()).decode() for p in r.rglob('*') if p.is_file() and (p.parent==r or p.parent.name in ('attempt01','attempt02') or 'runtime_logs' in p.parts) and p.suffix in ('.json','.log','.maps','.exit','.py')}))")
if p.returncode:raise RuntimeError(p.stderr)
for n,b in json.loads(p.stdout).items():
 q=R/n;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(base64.b64decode(b))
print('synced')
