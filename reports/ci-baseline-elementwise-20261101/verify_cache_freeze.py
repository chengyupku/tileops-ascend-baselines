"""Read-only final check that no remote cache inputs changed after audit."""
import json
from pathlib import Path
import shlex
import subprocess
import sys

server = sys.argv[1]
assert server in ('server1','server2')
base = Path(__file__).resolve().parent
target, root = {
    'server1':('openlibing-DevEnv-661495','/opt/tileops-baseline-cache/workspace'),
    'server2':('openlibing-DevEnv-055409','/opt/tileops-all-pairs/workspace'),
}[server]
code = '''import json,hashlib,sys
from pathlib import Path
data=json.load(sys.stdin);results=[]
for campaign,e in data['expected'].items():
 p=Path(data['root'])/'reports'/campaign/data['server']/'CACHE.jsonl'
 raw=p.read_bytes();rows=len(raw.splitlines());digest=hashlib.sha256(raw).hexdigest()
 results.append(dict(campaign=campaign,rows=rows,sha256=digest,unchanged=rows==e['rows'] and digest==e['sha256']))
print(json.dumps(dict(status='passed' if all(r['unchanged'] for r in results) else 'changed',catalogs=results)))
'''
data = dict(root=root,server=server,expected=json.loads((base/f'{server}-expected-cache-files.json').read_text()))
run = subprocess.run(['ssh',target,'python3 -c '+shlex.quote(code)],
                     input=json.dumps(data),text=True,capture_output=True,check=True)
result = json.loads(run.stdout)
(base/f'BASELINE_INPUT_FREEZE_RECHECK_{server}.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(server=server,**result)))
assert result['status'] == 'passed'
