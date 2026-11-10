"""Apply the prepared helper-only fix after all frozen measurement work exits."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tarfile

BASE = Path('/opt/tileops-ci/server2-20261030/overnight-20261101')
ROOT = BASE/'ci-full-v3/workspace'
sys.path.insert(0, str(ROOT/'tools'))
spec = importlib.util.spec_from_file_location('controller', ROOT/'tools/ci_elementwise32.py')
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)
state = json.loads((ROOT.parent/'measurements/STATE.json').read_text())
assert state['status'] == 'passed' and state['completed'] == state['expected'] == 70
post = BASE/'export-after-full-01/PRIVATE_REPORT.json'
assert post.exists(), 'Wait for the old-source post-suite export to finish before changing source'
assert json.loads(post.read_text())['qualified_cases'] == 527
assert not (BASE/'before-eager-helper-fix.json').exists()
before = controller.fingerprint()
controller.write(BASE/'before-eager-helper-fix.json', before)
paths = [ROOT/'tools/elementwise_eager_check.py',
         ROOT/'TileOPs/scripts/ci/elementwise32/payload/tools/elementwise_eager_check.py']
assert paths[0].read_bytes() == paths[1].read_bytes()
old = paths[0].read_text()
expected = old.replace('from workloads.numerics import compare_outputs',
                       'from workloads.numerics import Custom, compare_outputs')
expected = expected.replace('        check = compare_outputs(cpu(actual), cpu(reference), case.verification)', '''        evidence = case.verification
        if isinstance(evidence, Custom):
            # Custom validators can close over the supplied inputs. Rebind the
            # unchanged workload policy to the same input values on CPU.
            rebound = case.workload.verification(*cpu(inputs))
            assert isinstance(rebound, Custom) and rebound.reason == evidence.reason
            assert rebound.controls == evidence.controls and rebound.probe == evidence.probe
            evidence = rebound
        check = compare_outputs(cpu(actual), cpu(reference), evidence)''')
prepared = (BASE/'PREPARED_elementwise_eager_check.py').read_text()
assert expected == prepared and old != expected
with tarfile.open(BASE/'EAGER_HELPER_SOURCE_BEFORE.tar.gz', 'x:gz') as archive:
    for i,path in enumerate(paths):
        archive.add(path, arcname=f'helper-{i}.py')
for path in paths:
    path.write_text(prepared)
after = controller.fingerprint()
changes = {k for k in before if before[k] != after[k]}
assert before.keys() == after.keys() and changes == {str(p) for p in paths}
proof = dict(status='passed', files={k:dict(before=before[k],after=after[k]) for k in sorted(changes)},
             change='Rebind unchanged Custom workload verification to identical input values on CPU, preserving type/reason/controls/probe; Exact path unchanged.',
             production_changes=[], timing_unchanged=True, workload_policy_unchanged=True,
             whole_source_set_unchanged=True,
             prepared_source_sha256=hashlib.sha256(prepared.encode()).hexdigest())
controller.write(BASE/'EAGER_HELPER_SOURCE_PROOF.json', proof)
print(json.dumps(proof, indent=2))
