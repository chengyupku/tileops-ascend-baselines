"""Collect final evidence without loading Torch or touching the NPU."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OPS = ('LogFwdOp', 'Log1pFwdOp', 'SqrtFwdOp', 'RsqrtFwdOp')


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def receipt(label):
    rows = [read(p) for p in (HERE / 'requests').glob(label + '_*/status.json')]
    return max(rows, key=lambda r: r['completed_at']) if rows else None


current = read(HERE / 'CURRENT-FREEZE.json') or {'file': 'frozen-v3.json', 'selected': 'selected-configs-v3.json'}
frozen = read(HERE / current['file'])
source_matches = {
    name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
    for name, expected in (frozen or {}).get('sources', {}).items()
}
helper_hash = hashlib.sha256(
    (ROOT / 'TileOPs/src/tileops/kernels/ascend/elementwise/_logroots.py').read_bytes()
).hexdigest()
report = {
    'frozen_manifest': current['file'],
    'sources_match_frozen': source_matches,
    'selected_configurations': len(read(HERE / current['selected']) or []),
    'operators': {},
}
for op in OPS:
    kind = {'LogFwdOp': 'log', 'Log1pFwdOp': 'log1p', 'SqrtFwdOp': 'sqrt', 'RsqrtFwdOp': 'rsqrt'}[op]
    module_hash = hashlib.sha256(
        (ROOT / 'TileOPs/src/tileops/kernels/ascend/elementwise' / (kind + '.py')).read_bytes()
    ).hexdigest()
    edge_choices = []
    for path in (HERE / 'edges').glob('edges-' + op + '-*.json'):
        value = read(path)
        guard = receipt(path.stem)
        if (guard and guard['guard_exit_code'] == 0
                and value.get('helper_sha256') == helper_hash
                and value.get('module_sha256') == module_hash):
            edge_choices.append((guard['completed_at'], path, value, guard))
    _, edge_path, edge, edge_guard = max(edge_choices, key=lambda r: r[0]) if edge_choices else ('', None, None, None)
    checks = edge.get('checks', []) if edge else []
    pytest_choices = []
    for path in (HERE / 'edges').glob('pytest-' + op + '-*.xml'):
        guard = receipt(path.stem)
        if guard:
            pytest_choices.append((guard['completed_at'], path, guard))
    _, pytest_path, pytest_guard = max(pytest_choices, key=lambda r: r[0]) if pytest_choices else ('', None, None)
    tests = []
    if pytest_path is not None:
        for case in ET.parse(pytest_path).getroot().iter('testcase'):
            problem = case.find('failure')
            if problem is None:
                problem = case.find('error')
            tests.append({
                'name': case.get('name'),
                'passed': problem is None and case.find('skipped') is None,
                'problem': problem.get('message') if problem is not None else None,
            })
    profile_choices = []
    for path in (HERE / 'profiles').glob(op + '-*/16M-float32.json'):
        value = read(path)
        guard = receipt('profile-' + path.parent.name)
        if (guard and guard['guard_exit_code'] == 0
                and value.get('helper_sha256') == helper_hash
                and value.get('source_sha256') == module_hash):
            profile_choices.append((guard['completed_at'], path, value, guard))
    _, profile_path, profile, profile_guard = max(profile_choices, key=lambda r: r[0]) if profile_choices else ('', None, None, None)
    report['operators'][op] = {
        'edges': {
            'evidence': str(edge_path.relative_to(HERE)) if edge_path else None,
            'count': len(checks),
            'qualified': bool(edge and edge_guard and edge_guard['guard_exit_code'] == 0
                              and edge.get('helper_sha256') == helper_hash
                              and edge.get('module_sha256') == module_hash
                              and len(checks) == (33 if op == 'LogFwdOp' else 30)
                              and all(c['passed'] for c in checks)),
        },
        'pytest': {'tests': tests, 'guard_exit_code': pytest_guard and pytest_guard['guard_exit_code'],
                   'evidence': str(pytest_path.relative_to(HERE)) if pytest_path else None},
        'profile': {
            'evidence': str(profile_path.relative_to(HERE)) if profile_path else None,
            'qualified': bool(profile and profile_guard and profile_guard['guard_exit_code'] == 0
                              and profile.get('helper_sha256') == helper_hash
                              and profile.get('source_sha256') == module_hash
                              and profile.get('warmup') == 3 and profile.get('repeats') == 15
                              and profile.get('event_used') is False),
        },
    }
audit = read(HERE / 'audit.json') or {}
report['performance'] = {key: audit.get(key) for key in (
    'own_valid_cases', 'formal_valid_baseline_pairs', 'additional_valid_baseline_pairs',
    'expected_baseline_pairs', 'cases_passing', 'cases_passing_including_additional',
    'minimum_ratio', 'minimum_ratio_including_additional',
    'baseline_pairs_unattempted', 'baseline_pairs_without_valid_measurement',
)}
report['note'] = 'Evidence collection only. Test failures and unavailable baseline routes require explicit review; this report does not declare acceptance.'
(HERE / 'final-review.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
