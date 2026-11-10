"""CPU-only comparison with frozen CI manifest metadata; never materialize data."""
import hashlib
import json
from pathlib import Path
import sys
import types

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1] / '.worktrees/TileOPs-elementwise-ci-20261101'
PACKAGE = REPO / 'src/tileops'
pkg = types.ModuleType('tileops')
pkg.__path__ = [str(PACKAGE)]
sys.modules['tileops'] = pkg
from tileops.manifest import load_adts, load_manifest, load_workloads
from tileops.manifest.plan import entry_plan
from tileops.manifest.values import ADTValue
from tileops.manifest.workload import instantiate


def hashes():
    return {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (PACKAGE / 'manifest').rglob('*')
            if p.is_file() and p.suffix in {'.py', '.yaml'}}


def parameter(v):
    # Match benchmarks.api._report_value exactly, without importing torch.
    return {v.kind: dict(v.fields)} if isinstance(v, ADTValue) else v


before = hashes()
ops, adts = load_manifest(), load_adts()
snapshots = {}
for i, (name, entry) in enumerate(ops.items(), 1):
    plan = entry_plan(name, entry, adts, resolve=False)
    for row in load_workloads(name):
        for dtype in row.get('dtype_cases') or [{}]:
            call = instantiate(plan, row, dtype)
            key = (name, call.case_id)
            assert key not in snapshots
            tensors = call.tensors
            snapshot = dict(op=name, case_id=call.case_id, family=entry.get('family'),
                            tensors=tensors,
                            params=tensors | {k:parameter(v) for k,v in call.params.items()})
            snapshots[key] = json.loads(json.dumps(snapshot, default=str))
    if i % 25 == 0:
        print(json.dumps(dict(operators=i, cases=len(snapshots))), flush=True)
assert hashes() == before and 'torch' not in sys.modules
rows = json.loads((OUT / 'BASELINE32_PRIVATE.json').read_text())['rows']
matched, legacy, mismatches = 0, [], []
for row in rows:
    key = tuple(row['key'][1:])
    if key not in snapshots:
        legacy.append(row['key'])
    elif row['case'] != snapshots[key]:
        mismatches.append(dict(key=row['key'], cached=row['case'], current=snapshots[key]))
    else:
        matched += 1
report = dict(status='passed' if not mismatches else 'incompatible_contracts',
              cpu_only=True, operators=len(ops), manifest_cases=len(snapshots),
              exact_current_pairs=matched, legacy_pairs=legacy, mismatches=mismatches,
              sources=before)
(OUT / 'CURRENT_MANIFEST_CACHE_COMPATIBILITY.json').write_text(json.dumps(report,indent=2)+'\n')
(OUT / 'FROZEN_MANIFEST_SNAPSHOTS.json').write_text(json.dumps(
    dict(sources=before, cases=list(snapshots.values())),separators=(',',':'))+'\n')
counts = json.loads((OUT / 'FROZEN_MANIFEST_COUNTS.json').read_text())
assert counts['manifest_cases'] == len(snapshots)
counts['manifest_sources'] = before
(OUT / 'FROZEN_MANIFEST_COUNTS.json').write_text(json.dumps(counts,indent=2)+'\n')
print(json.dumps(dict(status=report['status'], exact_current_pairs=matched,
                     legacy_pairs=len(legacy), mismatches=len(mismatches))),flush=True)
