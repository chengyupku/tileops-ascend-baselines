"""Archive final CI raw profiles, immutable receipts, and exact source on CPU."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

BASE = Path('/opt/tileops-ci/server2-20261030/overnight-20261101')
JOB = BASE/'ci-full-v3'
REPO = JOB/'workspace/TileOPs'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


report = json.loads((BASE/'export-final-01/PRIVATE_REPORT.json').read_text())
assert report['qualified_operators'] == 70 and report['qualified_cases'] == 527
assert not report['missing']
assert json.loads((BASE/'export-final-01/INDEPENDENT_SNAPSHOT_AUDIT.json').read_text())['status'] == 'passed'
assert json.loads((BASE/'public-regressions-final-01/receipt.json').read_text())['status'] == 'passed'
destination = BASE/'ELEMENTWISE32_FINAL_EVIDENCE.tar.gz'
assert not destination.exists()
inputs = []


def add_tree(path, prefix):
    for item in sorted(path.rglob('*')):
        if '__pycache__' in item.parts or item.suffix == '.pyc' or item.is_symlink():
            continue
        if item.is_file():
            inputs.append((item, prefix+'/'+str(item.relative_to(path))))


add_tree(JOB, 'ci-full-v3')
for name in ['public-regressions-final-01', 'eager-requalification-01', 'export-final-01',
             'baseline-current-contract-audit']:
    add_tree(BASE/name, name)
add_tree((BASE/'current-baseline').resolve(), 'current-baseline')
for name in ['EAGER_HELPER_SOURCE_PROOF.json', 'EAGER_HELPER_SOURCE_BEFORE.tar.gz',
             'toolchain-v3.json',
             'EAGER_HELPER_READMISSION.json', 'PUBLIC_ORACLE_SOURCE_PROOF.json',
             'TEST_ONLY_READMISSION.json', 'PUBLIC_TEST_SOURCE_BEFORE.tar.gz',
             'PUBLIC_VALIDATION_EVIDENCE.tar.gz', 'PUBLIC_VALIDATION_EVIDENCE.json',
             'before-eager-helper-fix.json', 'export_verified_results.py',
             'requalify_dropout_eager.py', 'run_eager_requalification.py',
             'readmit_eager_helper.py', 'apply_eager_helper_fix.py',
             'audit_published_snapshot.py', 'archive_final_evidence.py']:
    inputs.append((BASE/name, name))
tracked = subprocess.check_output(['git','-C',str(REPO),'ls-files','-z']).decode().split('\0')
for name in tracked:
    if name and (REPO/name).is_file() and not (REPO/name).is_symlink():
        inputs.append((REPO/name, 'source/'+name))
assert len(inputs) == len({name for _,name in inputs})
included = {path.resolve() for path,_ in inputs}
windows = 0
for entry in json.loads((JOB/'workspace/ELEMENTWISE_PLAN.json').read_text()):
    directory = JOB/'measurements'/entry['op']
    if (directory/'LATEST_ATTEMPT.json').exists():
        directory = JOB/'measurements'/json.loads((directory/'LATEST_ATTEMPT.json').read_text())['path']
    for row in map(json.loads, (directory/'measurements.jsonl').read_text().splitlines()):
        for timing in (row, row['eager']):
            assert len(timing['retained_profiles']) == 15
            for trace in timing['retained_profiles']:
                profile = Path(trace['profile_path'])
                assert profile.is_dir() and profile.resolve().is_relative_to(JOB.resolve())
                required = [p for p in profile.rglob('*') if p.is_file()
                            and not p.name.endswith(('.done','.complete'))
                            and any(token in p.name for token in
                                    ('ffts_profile','hash_dic.slice_','task_track.slice_'))]
                assert required and all(p.resolve() in included for p in required)
                windows += 1
assert windows == 527*30 == 15810
manifest = {}
with tarfile.open(destination, 'x:gz', compresslevel=1) as archive:
    for path, name in inputs:
        digest = sha(path)
        archive.add(path, arcname=name, recursive=False)
        assert sha(path) == digest, 'Evidence changed during archival: '+str(path)
        manifest[name] = dict(bytes=path.stat().st_size, sha256=digest)
    metadata = dict(files=manifest, source_revision=report['private_source_revision'],
                    snapshot_sha256=report['snapshot_sha256'],
                    original_records_and_traces_preserved=True)
    raw = (json.dumps(metadata, separators=(',', ':'))+'\n').encode()
    header = tarfile.TarInfo('ARCHIVE_MANIFEST.json')
    header.size = len(raw)
    archive.addfile(header, io.BytesIO(raw))
summary = dict(status='passed', files=len(manifest), bytes=destination.stat().st_size,
               sha256=sha(destination), archive=str(destination), source=report['private_source_revision'],
               retained_profiler_windows=windows)
destination.with_suffix('.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(summary), flush=True)
