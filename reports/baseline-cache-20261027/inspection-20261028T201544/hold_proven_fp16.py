"""Exact temporary scheduling holds. CPU-only; baseline sources never changed."""
import fcntl
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, 'tools')
from run_baseline_cache import assigned_history, completed_ids, quarantine_key

host = sys.argv[1]
assert host in ('local', 'server1', 'server2')
root = Path('reports/baseline-cache-20261027')
hd = root / host
out = root / 'inspection-20261028T201544'
read = lambda p: json.loads(p.read_text())
probe = read(out / (host + '-FP16_ROUTES.json'))
tasks = {t['id']: t for t in read(root / 'PLAN.json')['pairs'] if t['host'] == host}
sources = {}
if host == 'local':
    p = Path('TileOPs/benchmarks/ascend_all_pairs/ops_transformer/adapters.py')
    assert "if case.inputs[0].dtype!=torch.bfloat16:" in p.read_text()
    assert 'RecurrentKda native Q/K/V are BF16; FP16 cannot be narrowed losslessly.' in p.read_text()
    sources[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    reason = 'Exact current FP16 GDN route reaches recurrent_family, whose original interface requires BF16. CPU dispatch proved without importing device runtimes; no lossy cast permitted. Temporarily hold before device startup; not a new numerical failure or successful cache.'
else:
    base = Path('ascend_baselines/sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/fla')
    for name, expected in {
        'cumsum.py': '746c5bb3c86fc513ccba844524e9d01cf10c47d7718288e1dae74e0a9490b690',
        'chunk.py': 'fdb3cd8654e664b4b2734c504c7bf5a2cb1e975596cd97a7762a8ac95936205c',
    }.items():
        p = base / name
        assert hashlib.sha256(p.read_bytes()).hexdigest() == expected
        sources[str(p)] = expected
    reason = 'Exact FP16 GDN current native dispatch selects original SGL Triton path. Original chunk.py calls cumsum with default FP32 output; cumsum.py stores s.dtype FP16 to that FP32 block pointer. Repeated server1 full-case compile failures prove this mismatch; server2 source hashes identical. Temporary exact-case hold inferred from shared path, not an executed failure for this case. No kernel/compiler/checker changes.'

added = []
excluded = []
with (hd / 'QUARANTINE.lock').open('a+') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    q = read(hd / 'QUARANTINE.json')
    prior = out / (host + '-QUARANTINE.before.json')
    assert not prior.exists()
    prior.write_text(json.dumps(q, ensure_ascii=False, indent=2) + '\n')
    rows = [json.loads(s) for s in (hd / 'journal.jsonl').read_text().splitlines() if s.strip()]
    cache = [json.loads(s) for s in (hd / 'CACHE.jsonl').read_text().splitlines() if s.strip()]
    latest, cached = assigned_history(list(tasks.values()), rows, cache)
    done = completed_ids(latest, read(hd / 'RETRY.json'), q, cached)
    live = read(hd / 'LIVE.json')
    for entry in probe['definite_cpu_admission_blocks']:
        t = tasks[entry['id']]
        assert t['key'][1] == 'GDNFwdOp' and t['case']['tensors']['q'][1] == 'float16'
        assert t['case']['tensors']['g'][1] == 'float16'
        assert t['key'][0] == ('ops-transformer' if host == 'local' else 'sgl-kernel-npu')
        if t['id'] in done or quarantine_key(t, q) or t['key'] == live.get('key'):
            excluded.append({'id': t['id'], 'reason': 'cached/terminal/already-held/current-pair'})
            continue
        q['pair:' + t['id']] = {
            'reason': reason, 'at': time.time(), 'key': t['key'],
            'inspection': 'inspection-20261028T201544',
            'evidence': [str(out / (host + '-FP16_ROUTES.json')), str(out / (host + '-analysis.json'))],
            'source_sha256': sources,
            'case_sha256': hashlib.sha256(json.dumps(t['case'], sort_keys=True).encode()).hexdigest(),
            'release_condition': 'Reviewed actual route/deployment change that resolves this exact interface/compilation limitation; preserve original implementation and full checker; record RETRY revision.',
            'not_numerical_verdict': True,
        }
        added.append({'id': t['id'], 'key': t['key']})
    tmp = hd / 'QUARANTINE.inspection-201544.tmp'
    tmp.write_text(json.dumps(q, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, hd / 'QUARANTINE.json')
result = {'at': time.time(), 'host': host, 'added': added, 'count': len(added), 'excluded': excluded,
          'reason': reason, 'source_sha256': sources, 'current_pair_untouched': live.get('key')}
(out / (host + '-FP16_HOLDS.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k not in ('added', 'reason')}, ensure_ascii=False))
