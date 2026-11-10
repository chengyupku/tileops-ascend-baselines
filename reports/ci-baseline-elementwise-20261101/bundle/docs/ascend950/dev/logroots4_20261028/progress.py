"""Read-only CPU progress summary; never imports device libraries."""
import datetime
import argparse
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--detail',action='store_true');args=p.parse_args()
print(datetime.datetime.now(datetime.timezone.utc).isoformat())
active=[];completed=[]
for path in (HERE/'requests').glob('*'):
    status=path/'status.json'
    if status.exists():completed.append(json.loads(status.read_text()))
    elif (path/'guard.log').exists():
        text=(path/'guard.log').read_text()
        item={'request':path.name,'phase':'running' if '[npu-guard] acquired device' in text else 'queued'}
        if args.detail:item['last_output']=text[-1200:]
        active.append(item)
print(json.dumps({'completed_requests':len(completed),'active':active},indent=2))
for filename in ('complete-search-status.json','formal-batch-v1-status.json','baseline-tla-status.json','baseline-catlass-status.json','baseline-compile-status.json'):
    path=HERE/filename
    if path.exists():
        rows=json.loads(path.read_text())
        print(filename,'done',len(rows),'last',rows[-1]['label'],rows[-1]['rc'])
