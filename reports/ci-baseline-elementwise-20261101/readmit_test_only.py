"""Revalidate retained measurements after a documented public-test-only change.

This never admits production, workload, measurement or timing changes. All raw
profiles and original receipts are retained, and every window is replayed.
"""
import importlib.util
import json
from pathlib import Path
import sys
import time

BASE = Path('/opt/tileops-ci/server2-20261030/overnight-20261101')
ROOT = BASE / 'ci-full-v3/workspace'
OUT = BASE / 'ci-full-v3/measurements'
sys.path.insert(0, str(ROOT / 'tools'))
spec = importlib.util.spec_from_file_location('controller', ROOT / 'tools/ci_elementwise32.py')
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)

before = json.loads((BASE / 'before-public-source-fix.json').read_text())
after = controller.fingerprint()
assert before.keys() == after.keys(), 'Source set changed'
allowed = {str(ROOT / 'TileOPs/tests/ops/test_special_elementwise.py'),
           str(ROOT / 'TileOPs/tests/ops/test_binary_arith.py'),
           str(ROOT / 'TileOPs/tests/ops/test_elementwise_unary_activation_conformance.py'),
           str(ROOT / 'ELEMENTWISE_EXPORT.json'),
           str(ROOT / 'TileOPs/scripts/ci/elementwise32/payload/ELEMENTWISE_EXPORT.json')}
changes = {k for k in before if before[k] != after[k]}
assert changes and changes <= allowed, changes
assert str(ROOT / 'TileOPs/tests/ops/test_special_elementwise.py') in changes
semantic_proof = json.loads((BASE / 'PUBLIC_ORACLE_SOURCE_PROOF.json').read_text())
assert semantic_proof['status'] == 'passed' and semantic_proof['tolerances_unchanged'] is True
assert semantic_proof['before_sha256'] == before[str(ROOT / 'TileOPs/tests/ops/test_special_elementwise.py')]
assert semantic_proof['after_sha256'] == after[str(ROOT / 'TileOPs/tests/ops/test_special_elementwise.py')]
for name, proof in semantic_proof['files'].items():
    assert proof['before_sha256'] == before[str(ROOT / 'TileOPs' / name)]
    assert proof['after_sha256'] == after[str(ROOT / 'TileOPs' / name)]
    assert proof['all_assertions_unchanged'] and proof['unmodified_module_ast_equal']
plan = json.loads((ROOT / 'ELEMENTWISE_PLAN.json').read_text())
admitted = []
prepared = []
for entry in plan:
    directory = OUT / entry['op']
    pointer = directory / 'LATEST_ATTEMPT.json'
    if pointer.exists():
        directory = OUT / json.loads(pointer.read_text())['path']
        assert directory.resolve().is_relative_to(OUT.resolve())
    receipt_path = directory / 'receipt.json'
    if not receipt_path.exists():
        continue
    receipt = json.loads(receipt_path.read_text())
    if receipt['status'] != 'passed':
        continue
    previous_path = directory / 'READMISSION.json'
    previous_digest = None
    previous_bytes = previous_path.read_bytes() if previous_path.exists() else None
    if receipt['sources'] != before:
        previous = json.loads(previous_path.read_text())
        assert previous['status'] == 'passed' and previous['current_sources'] == before
        assert previous['original_receipt_sha256'] == controller.sha(receipt_path)
        assert previous['production_kernel_changes'] == []
        previous_digest = controller.sha(previous_path)
    elif previous_path.exists():
        raise AssertionError('Unexpected redundant proof; preserve and inspect it')
    # The saved guard certifies this original complete device command, not a
    # new device run. A test-only edit cannot create a new device measurement.
    guard = json.loads((directory / 'guard.json').read_text())
    assert guard['phase'] == 'finished' and guard['workload_exit_code'] == 0
    assert guard['performance_valid'] is True
    assert not guard.get('conflict_detected') and not guard.get('visibility_lost')
    if 'command' in receipt:
        assert guard['owner'] == receipt['command'][receipt['command'].index('--owner')+1]
    else:
        assert entry['op'] == 'LogicalNotFwdOp' and receipt.get('recovered_after_controller_pause') is True
        assert controller.sha(receipt_path) == '12129db6132bc29d8fe756a6c79dc93ba4d64f7df6147f362506687cf6c762b3'
        assert guard['owner'] == 'ci/elementwise32/measurements/LogicalNotFwdOp'
    for name, digest in receipt['sources'].items():
        if '/src/tileops/kernels/' in name or '/src/tileops/ops/' in name:
            assert after[name] == digest, 'Production changed: ' + name
    rows = [json.loads(l) for l in (directory / 'measurements.jsonl').read_text().splitlines()]
    for row in rows:
        for name, digest in row['source_manifest'].items():
            assert controller.sha(name) == digest, 'Measured dependency changed: ' + name
    controller.validate_rows(rows, entry['case_ids'])
    proof = dict(status='passed', at=time.time(), op=entry['op'],
                 reason='Only public Ascend reference/marker tests and export hashes changed; original assertions and tolerances retained',
                 current_sources=after, original_receipt_sha256=controller.sha(receipt_path),
                 original_guard_sha256=controller.sha(directory / 'guard.json'),
                 production_kernel_changes=[], measured_dependency_changes=[],
                 previous_readmission_sha256=previous_digest,
                 changes={name:dict(before=before[name], after=after[name]) for name in sorted(changes)},
                 semantic_proof_sha256=controller.sha(BASE / 'PUBLIC_ORACLE_SOURCE_PROOF.json'),
                 cases=len(rows), replayed_windows=len(rows)*30)
    prepared.append((previous_path, proof, previous_bytes, previous_digest))
    admitted.append(dict(op=entry['op'], cases=len(rows), replayed_windows=len(rows)*30))
assert controller.fingerprint() == after
# Validate every op before changing any admission pointer.
for path, proof, previous_bytes, previous_digest in prepared:
    assert (path.read_bytes() if path.exists() else None) == previous_bytes
    if previous_digest is not None:
        backup = path.with_name(f'READMISSION.{previous_digest}.json')
        if backup.exists():
            assert backup.read_bytes() == previous_bytes
        else:
            backup.write_bytes(previous_bytes)
    controller.write(path, proof)
controller.write(BASE / 'TEST_ONLY_READMISSION.json', dict(status='passed', changes=sorted(changes),
                 production_kernel_changes=[], admitted=admitted,
                 total_cases=sum(r['cases'] for r in admitted),
                 replayed_windows=sum(r['replayed_windows'] for r in admitted)))
print(json.dumps(dict(status='passed', operators=len(admitted),
                     cases=sum(r['cases'] for r in admitted),
                     replayed_windows=sum(r['replayed_windows'] for r in admitted))))
