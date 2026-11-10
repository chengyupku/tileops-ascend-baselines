"""Read-only reassessment of the catalog's historical-retained cohort.

This writes assessment artifacts only: it does not promote catalog entries,
change runtime code, or execute/import any baseline or device library.
"""
import collections
import concurrent.futures
import csv
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

W = Path('/root/dyq/workspace')
OUT = Path(__file__).resolve().parent
CAT = W / '.worktrees/TileOPs-baseline-catalog/docs/ascend950/baselines'
FULL = W / '.cache/all-pairs/full_rollout_20261024'
STAGE = W / '.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/rollout_staging'
sources = {}


def read(path):
    path = Path(path)
    b = path.read_bytes()
    sources[str(path)] = {'sha256': hashlib.sha256(b).hexdigest(), 'bytes': len(b)}
    return json.loads(b)


def read_lines(path):
    path = Path(path)
    b = path.read_bytes()
    sources[str(path)] = {'sha256': hashlib.sha256(b).hexdigest(), 'bytes': len(b)}
    return [json.loads(x) for x in b.splitlines() if x.strip()]


def dump(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def keyset(d):
    return {tuple(k) for k in d.get('keys', []) if isinstance(k, list) and len(k) == 3}


catalog = [json.loads(x) for x in gzip.open(CAT / 'observed_pairs.jsonl.gz', 'rt')]
H = {tuple(r['key']) for r in catalog if r['status'] == 'historical_retained'}
snapshot = {tuple(r['key']): r for r in read_lines(W / 'reports/baseline-status-20261026/EXACT_TARGET_SETS.snapshot.jsonl') if tuple(r['key']) in H}
history = read(STAGE / 'historical_checkpoint.json')
old = keyset(history)
validated = [r for r in read(FULL / 'dashboard/VALIDATED_HISTORY.json')['rows'] if tuple(r['key']) in H]
validated_index = {tuple(r['key']): r for r in validated}

# Verify the immutable links of the numerical acceptance chain. Full independent
# reviews are reused as historical evidence; not claimed to be newly repeated.
links = []
acceptances = collections.defaultdict(list)
visited = set()


def linked(ref, default_hash=None):
    if isinstance(ref, str):
        ref = {'path': ref, 'sha256': default_hash}
    p = Path(ref['path'])
    entry = {'path': str(p), 'expected_sha256': ref.get('sha256')}
    if not p.is_file():
        entry['status'] = 'missing'
        links.append(entry)
        return None
    d = read(p)
    entry['actual_sha256'] = sources[str(p)]['sha256']
    entry['status'] = 'hash_match' if ref.get('sha256') == entry['actual_sha256'] else ('unsealed' if not ref.get('sha256') else 'hash_mismatch')
    entry['cohort_keys'] = len(keyset(d) & H) if isinstance(d, dict) else 0
    links.append(entry)
    return d


def walk_progress(path):
    if path in visited:
        return
    visited.add(path)
    d = read(path)
    refs = d.get('root_acceptances', [])
    if isinstance(refs, dict):
        refs = list(refs.values())
    if d.get('root_acceptance'):
        refs = list(refs) + [d['root_acceptance']]
    for ref in refs:
        a = linked(ref, d.get('root_acceptance_sha256'))
        if isinstance(a, dict):
            p = ref if isinstance(ref, str) else ref['path']
            for k in keyset(a) & H:
                acceptances[k].append({'path': p, 'status': a.get('status'), 'link_status': links[-1]['status']})
    prev = d.get('prior_progress') or d.get('prior_progress_path')
    if prev:
        p = prev if isinstance(prev, str) else prev['path']
        linked(prev, d.get('prior_progress_sha256'))
        walk_progress(p)
    for ref in d.get('local_source_documents', []):
        linked(ref)


walk_progress(history['source_checkpoint'])
for ref in history['sources']:
    rows = read_lines(ref['snapshot'])
    links.append({'path': ref['snapshot'], 'expected_sha256': ref['sha256'], 'actual_sha256': sources[ref['snapshot']]['sha256'], 'status': 'hash_match' if sources[ref['snapshot']]['sha256'] == ref['sha256'] else 'hash_mismatch'})
    for i, r in enumerate(rows):
        ks = keyset(r)
        if isinstance(r.get('key'), list):
            ks.add(tuple(r['key']))
        for k in ks & H:
            acceptances[k].append({'path': ref['snapshot'], 'line': i + 1, 'status': r.get('decision', r.get('event')), 'link_status': links[-1]['status']})

# Same CPU-only reader is run locally and over each existing SSH host. It returns
# compact checker fields, never tensors or executable code from the baseline.
READER = r'''
import json,hashlib,pathlib
paths = PATHS
out=[]
for p in paths:
 r={'path':p}
 try:
  b=pathlib.Path(p).read_bytes(); d=json.loads(b)
  r.update(sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),key=[d.get('library'),d.get('op'),d.get('case_id')])
  for field in ['status','verification','inputs_preserved','provider_source_changed','oracle_source_changed','api','adaptation','limitations','case_snapshot','independent_cpu_reference']:
   r[field]=d.get(field)
  calls=d.get('observed_calls',{}).get('candidate',[])
  r['candidate_returned']=bool(calls) and all(x.get('returned') is True for x in calls)
  r['source_bindings_count']=len(d.get('source_sha256') or {})
  r['provider_bindings_count']=len(d.get('provider_source_sha256') or {})
 except Exception as e:r['read_error']=str(e)
 out.append(r)
print(json.dumps(out,ensure_ascii=False))
'''


def checker_read(host, rows):
    paths = sorted({r['checker'] for r in rows if r.get('checker')})
    script = READER.replace('PATHS', repr(paths))
    if host == 'local':
        cmd = ['python3', '-']
    else:
        alias = {'server1': 'openlibing-DevEnv-661495', 'server2': 'openlibing-DevEnv-055409'}[host]
        cmd = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12', alias, 'python3 -']
    p = subprocess.run(cmd, input=script, text=True, capture_output=True, timeout=55)
    if p.returncode:
        return {'host': host, 'error': p.stderr[-2000:], 'rows': []}
    return {'host': host, 'rows': json.loads(p.stdout)}


hosts = sorted({r['host'] for r in validated})
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    results = list(pool.map(lambda host: checker_read(host, [r for r in validated if r['host'] == host]), hosts))
checks = {r['path']: r for group in results for r in group['rows']}
dump('CHECKER_READS.json', results)

rows = []
for k in sorted(H):
    s = snapshot[k]
    flags = []
    if s.get('modified_dependency_receipts_present'):
        flags.append('modified_dependency_receipt')
    if s['substatus'] != 'historical_success_original_provenance_unresolved':
        flags.append(s['substatus'])
    v = validated_index.get(k)
    c = checks.get(v['checker']) if v else None
    r = {'key': list(k), 'old_status': 'historical_retained', 'prior_reason': s['substatus'], 'numeric_historical_acceptance': k in old, 'numeric_checker_index': bool(v), 'boundary_flags': flags, 'historical_acceptance_links': acceptances[k], 'checker': v, 'checker_read': c, 'old_evidence': s['scope1529_evidence'], 'flag_evidence': s.get('refresh_evidence', {}), 'boundary_evidence': s.get('evidence', {})}
    if flags:
        r['assessment'] = 'specific_boundary_review'
    elif k not in old and not v:
        r['assessment'] = 'additional_receipt_binding_needed'
    else:
        r['assessment'] = 'restoration_candidate_numeric_evidence_no_recorded_boundary_conflict'
    rows.append(r)

(OUT / 'PAIRS.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
counts = collections.Counter(r['assessment'] for r in rows)
summary = {
    'at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'scope': len(H), 'historical_acceptance_count': len(H & old),
    'checker_index_count': len(validated_index),
    'numeric_union_count': len(H & (old | set(validated_index))),
    'assessment_counts': dict(counts),
    'candidate_by_library': dict(collections.Counter(r['key'][0] for r in rows if r['assessment'].startswith('restoration_candidate'))),
    'checker_reads_by_host': {r['host']: {'read': len(r['rows']), 'error': r.get('error')} for r in results},
    'checker_statuses': dict(collections.Counter(str(r.get('status', r.get('read_error'))) for r in checks.values())),
    'acceptance_link_checks': dict(collections.Counter(r['status'] for r in links)),
    'direct_linked_historical_acceptance_count': sum(bool(acceptances[k]) for k in old & H),
    'catalog_changed': False, 'npu_runs': 0,
    'interpretation': 'Candidate is not a completed original-boundary audit. Reuse prior exact numerical acceptance; do not call an old generic missing entire compiler closure finding a proven internal modification. No blanket promotion from user policy alone.',
}
dump('SUMMARY.json', summary)
dump('ACCEPTANCE_LINKS.json', links)
dump('SOURCE_HASHES.json', sources)
print(json.dumps(summary, ensure_ascii=False, indent=2))
