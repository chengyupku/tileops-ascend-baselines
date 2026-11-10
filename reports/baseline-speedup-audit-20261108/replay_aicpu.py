"""CPU-only replay of every flagged AICPU measurement, without rerunning devices."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile
import tempfile

P = Path(__file__).resolve().parent
rows = json.loads((P / 'ROWS.json').read_text())
index = json.loads((P / 'EVIDENCE_INDEX.json').read_text())
path = P.parent / 'torch-all2208-20261105/tools/kernel_task_profile_full32.py'
spec = importlib.util.spec_from_file_location('raw_parser', path)
parser = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parser)
results = []
for row in rows:
    if 'aicpu_present' not in row['flags']:
        continue
    entry = index[row['measurement_id']]
    archive = Path(entry['evidence']['archive'])
    with archive.open('rb') as f:
        assert hashlib.file_digest(f, 'sha256').hexdigest() == entry['evidence']['archive_sha256']
    measurement = json.loads(Path(row['evidence']).read_text())
    timing = measurement.get('timing', measurement)
    windows = []
    with tempfile.TemporaryDirectory(prefix='baseline-aicpu-replay-') as tmp:
        with tarfile.open(archive, 'r|gz') as tar:
            for member in tar:
                # Keep the binary timing/name records needed by the pure CPU decoder.
                if member.isfile() and str(Path(member.name)).startswith('profiles/') and any(
                    s in member.name for s in ['ffts_profile', 'stars_soc.data.', 'hash_dic.slice_', 'task_track.slice_', 'info.json']
                ):
                    tar.extract(member, tmp, filter='data')
        for recorded, sample in zip(timing['retained_profiles'], timing['latency_samples_ms'], strict=True):
            local = Path(tmp) / ('profiles/' + recorded['profile_path'].split('/profiles/', 1)[1])
            digest = parser.trace_digest(local)
            assert digest == recorded['profile_trace_sha256']
            parsed = parser.parse_profile(local, 1)
            assert abs(parsed['latency_ms'] - sample) < 1e-7
            cpu = [t for t in parsed['profile_kernel_tasks'] if 'AICPU' in t['core_roles']]
            windows.append({'latency_ms': parsed['latency_ms'], 'aicpu_count': len(cpu),
                            'aicpu_us': sum(t['duration_us'] for t in cpu), 'trace_sha256': digest})
    result = {'key': [row['library'], row['op'], row['case']], 'measurement_id': row['measurement_id'],
              'archive': str(archive), 'archive_sha256': entry['evidence']['archive_sha256'],
              'windows': windows, 'status': 'all_15_raw_windows_match'}
    results.append(result)
    (P / 'AICPU_RAW_REPLAY.json').write_text(json.dumps(results, indent=2) + '\n')
    print('replayed', len(results), result['key'], flush=True)
