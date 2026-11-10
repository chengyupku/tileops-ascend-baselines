import collections
import json
import sys
from pathlib import Path

root, host = Path(sys.argv[1]), sys.argv[2]
campaigns = [root / 'reports/baseline-cache-20261027']
if host == 'server1':
    campaigns.append(root / 'reports/baseline-old1076-20261029')
latest = {}
live = []
recent = []
for campaign in campaigns:
    directory = campaign / host
    p = directory / 'LIVE.json'
    if p.exists():
        d = json.loads(p.read_text())
        item = {k: d.get(k) for k in ('at', 'phase', 'key', 'out', 'pair_phase', 'counts', 'remaining')}
        out = Path(d.get('out', '/nonexistent'))
        for name in ('correctness.json', 'measurement.json', 'task.json'):
            p = out / name
            if p.exists():
                r = json.loads(p.read_text())
                item[name] = {k: r.get(k) for k in ('status', 'timing_mode', 'comparison_metric', 'entrypoint', 'repair_revision')}
        live.append(item)
    p = directory / 'CACHE.jsonl'
    if not p.exists():
        continue
    for line in p.open():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get('status') != 'cached' or r.get('performance_valid') is not True:
            continue
        value = {k: r.get(k) for k in ('key', 'comparison_metric', 'timing_mode', 'finished_at', 'correctness_receipt')}
        key = tuple(r['key'])
        if key not in latest or value['finished_at'] > latest[key]['finished_at']:
            latest[key] = value
        recent.append(value)
counts = collections.Counter(r['comparison_metric'] for r in latest.values())
libs = collections.defaultdict(collections.Counter)
for r in latest.values():
    libs[r['key'][0]][r['comparison_metric']] += 1
print(json.dumps(dict(host=host, counts=counts, libraries=libs, live=live,
                     recent=sorted(recent, key=lambda r: r['finished_at'])[-10:]), separators=(',', ':')))
