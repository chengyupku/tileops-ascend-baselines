"""Read-only evidence audit; emit a complete local32 development summary.

Does not claim acceptance against the pending dedicated 32-core baselines.
"""
import ast
import datetime
import hashlib
import json
from pathlib import Path

r = Path('/root/dyq/workspace/docs/ascend950/dev/elementwise_full_local32_20261030')
root = r.parents[3]
def read(p):
    return json.loads(Path(p).read_text())
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan = read(r / 'PLAN.json')
ledger = read(r.parent / 'elementwise_serial_20261030/STATE.json')
assert len(plan['remaining_scope']) == len(set(plan['remaining_scope'])) == 66
assert set(plan['local32_completed']) == set(plan['remaining_scope'])
assert len(plan['local32_completed']) == 66
assert len(ledger['operators']) == 70
assert {v['op'] for v in ledger['operators']} == set(plan['remaining_scope'] + plan['completed_pilot'])
assert all(v.get('local32_validated') for v in ledger['operators'])
reports = {}
for p in r.glob('*REPORT.json'):
    d = read(p)
    if 'op' in d:
        assert d['op'] not in reports
        reports[d['op']] = (p, d)
assert set(reports) == set(plan['remaining_scope'])
# These two sibling-only edits already have new-source public regressions.
# Recheck both hashes and the unaffected AST, rather than ignore source drift.
allowed_drift = {}
for sibling, old_op, filename in [
    ('FloorDivideFwdOp', 'RemainderFwdOp', 'arithmetic_modulo.py'),
    ('MinimumFwdOp', 'MaximumFwdOp', 'arithmetic_extrema.py'),
]:
    d = reports[sibling][1]['shared_source_scope']
    old = root / d['prior_source']
    new = root / 'TileOPs/src/tileops/kernels/ascend/elementwise' / filename
    assert sha(old) == d['before_sha256'] and sha(new) == d['after_sha256']
    class RemoveSibling(ast.NodeTransformer):
        def visit_ClassDef(self, node):
            return None if node.name == d['only_changed_ast_node'] else self.generic_visit(node)
    def normalized(path):
        return ast.dump(RemoveSibling().visit(ast.parse(path.read_text())), include_attributes=False)
    assert normalized(old) == normalized(new)
    allowed_drift[(old_op, str(new))] = d
rows = []
reference_notes = {}
for op in plan['remaining_scope']:
    p, d = reports[op]
    assert d['runtime_limit'] == {'cube': 32, 'vector': 64}, op
    assert d['manifest_correct'] == d['same_host_eager_80_percent_passes'] > 0, op
    ratio_keys = [k for k in d if k.endswith('eager_ratio')]
    assert len(ratio_keys) == 1, (op, ratio_keys)
    ratio = d[ratio_keys[0]]
    assert ratio >= .8, (op, ratio)
    assert d['full_registered_baseline_acceptance'] is False
    for path, digest in d['source_manifest'].items():
        if sha(path) != digest:
            exception = allowed_drift[(op, path)]
            assert digest == exception['before_sha256']
    rows.append(dict(op=op, scope='goal', manifest_cases=d['manifest_correct'], minimum_eager_ratio=ratio, report=str(p), report_sha256=sha(p)))
    notes = {k: v for k, v in d.items() if 'reference' in k or 'qualification' in k}
    if notes:
        reference_notes[op] = notes
pilot_path = r.parent / 'bitwise4_local32_20261030/FINAL_REPORT.json'
pilot = read(pilot_path)
assert pilot['hardware']['runtime_cube'] == 32 and pilot['hardware']['runtime_vector'] == 64
for path, digest in pilot['sources'].items():
    assert sha(root / path) == digest, path
for v in pilot['receipts'].values():
    g = read(root / v['receipt'])
    assert g['exit_code'] == 0 and g['performance_valid']
for v in pilot['operators']:
    assert v['correct'] == v['performance_passes'] == v['cases']
    assert v['minimum_eager_ratio'] >= .8
    measurements = [json.loads(line) for line in (root / v['evidence']).read_text().splitlines()]
    assert len(measurements) == v['cases']
    rows.append(dict(op=v['op'], scope='pilot', manifest_cases=v['cases'], minimum_eager_ratio=v['minimum_eager_ratio'], report=str(pilot_path), report_sha256=sha(pilot_path)))
assert len(rows) == 70
report = dict(
    completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    status='local32_development_validated_registered_baseline_pending',
    goal_operators=66, pilot_operators=4, total_operators=70,
    goal_manifest_cases=sum(v['manifest_cases'] for v in rows if v['scope']=='goal'),
    total_manifest_cases=sum(v['manifest_cases'] for v in rows),
    minimum_eager_ratio=min(v['minimum_eager_ratio'] for v in rows),
    runtime_limit={'cube':32,'vector':64}, physical_hardware={'cube':36,'vector':72},
    full_registered_baseline_acceptance=False,
    registered_baseline_pairs=sum(d['registered_baseline_pairs'] for _,d in reports.values())+pilot['registered_baseline_pairs'],
    qualification='Local physical36AIC/72AIV with process32AIC/64AIV. Same-host eager >=80% gate; dedicated remote32 baseline acceptance remains pending. See per-op reference notes and Sinusoidal original-NPU-reference discrepancy.',
    operators=rows, reference_notes=reference_notes,
    shared_source_equivalence=[dict(op=op,path=path,**d) for (op,path),d in allowed_drift.items()],
)
Path('/root/dyq/workspace/reports/ci-baseline-elementwise-20261101/LOCAL_SOURCE_AUDIT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['operators','reference_notes','shared_source_equivalence']},ensure_ascii=False,indent=2))
