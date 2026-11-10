"""Inspection-only scoped holds; no device execution or success promotion."""
import sys, json, time, os, tempfile, fcntl
from pathlib import Path

root = Path.cwd()
sys.path.insert(0, str(root / 'tools'))
from run_baseline_cache import assigned_history, completed_ids, quarantine_key

host = sys.argv[1]
assert host in ('local', 'server2')
campaign = root / 'reports/baseline-cache-20261027'
report = campaign / 'inspection-20261028T121544'
hostdir = campaign / host
candidate_file = report / ('LOCAL_GDN_INTERFACE_CANDIDATES.json' if host == 'local' else 'REMOTE_MISSING_PROVIDERS_CANDIDATES.json')
reason = (
    'Current default GDN FP16 adapter forwards FP16 gate/beta but selected original native binary requires FP32. '
    'Repeated identical OpDef failures; separately registered successful exact routes excluded. '
    'Temporary interface-contract hold, not a conclusion that FLA or all GDN is unusable.'
    if host == 'local' else
    'Current remote provider registry, exact PTO registrations and historical fallback have no factory for this operator. '
    'CPU registry preflight establishes this exact pair cannot execute with current deployment. '
    'Temporary missing-interface hold, not a whole-library failure.'
)

def lines(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]

def atomic(path, data):
    fd, name = tempfile.mkstemp(prefix=path.name+'.inspection-', dir=path.parent)
    with os.fdopen(fd, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(name, path)

with (hostdir / 'QUARANTINE.lock').open('a+') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    tasks = [t for t in json.loads((campaign/'PLAN.json').read_text())['pairs'] if t['host'] == host]
    owned = {t['id']: t for t in tasks}
    qpath = hostdir/'QUARANTINE.json'
    q = json.loads(qpath.read_text())
    before = report/(host+'-QUARANTINE.before.json')
    if not before.exists():
        with before.open('x') as f: json.dump(q, f, ensure_ascii=False, indent=2)
    latest, cached = assigned_history(tasks, lines(hostdir/'journal.jsonl'), lines(hostdir/'CACHE.jsonl'))
    done = completed_ids(latest, json.loads((hostdir/'RETRY.json').read_text()), q, cached)
    live = json.loads((hostdir/'LIVE.json').read_text())
    added = []
    for t in json.loads(candidate_file.read_text())['pairs']:
        assert t['id'] in owned and owned[t['id']]['key'] == t['key']
        if t['id'] in done or quarantine_key(t, q) or t['key'] == live.get('key'):
            continue
        q['pair:'+t['id']] = dict(reason=reason, key=t['key'], scope='exact_pair', temporary=True,
                                  at=time.time(), inspection='inspection-20261028T121544',
                                  evidence=str(candidate_file), release_condition='Recorded interface/deployment change and explicit RETRY revision; never rerun cached pairs')
        added.append(t['id'])
    atomic(qpath, q)
    action = dict(at=time.time(), host=host, added=added, count=len(added),
                  supervisor=live['identity'], current_key=live.get('key'),
                  cache_count_before=len(cached), reason=reason,
                  note='Verify supervisor reread after current pair; an in-flight failure may publish its older quarantine snapshot.')
    with (report/(host+'-hold-actions.jsonl')).open('a') as f:
        f.write(json.dumps(action, ensure_ascii=False)+'\n')
    print(json.dumps(action, ensure_ascii=False))
