"""Freeze original cache bytes and correctness/guard proofs without device access."""
import hashlib
import json
from pathlib import Path
import sys
import tarfile

root, host, expected_file, destination = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
expected = json.loads(expected_file.read_text())
seen = set()
count = 0
with tarfile.open(destination, 'w:gz', compresslevel=1) as archive:
    for campaign, specification in expected.items():
        cache = root / 'reports' / campaign / host / 'CACHE.jsonl'
        raw = cache.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == specification['sha256']
        rows = [json.loads(line) for line in raw.splitlines()]
        assert len(rows) == specification['rows']
        archive.add(cache, arcname='catalogs/' + campaign + '/CACHE.jsonl', recursive=False)
        for row in rows:
            assert row['cache_key'] not in seen
            seen.add(row['cache_key'])
            correctness = Path(row['correctness_receipt'])
            assert hashlib.sha256(correctness.read_bytes()).hexdigest() == row['correctness_sha256']
            guard = Path(row['guard_receipt'])
            binding = row.get('guard_binding')
            if binding:
                assert hashlib.sha256(guard.read_bytes()).hexdigest() == binding['receipt_sha256']
            prefix = 'proofs/' + row['cache_key'] + '/'
            archive.add(correctness, arcname=prefix + 'correctness.json', recursive=False)
            archive.add(guard, arcname=prefix + 'guard.json', recursive=False)
            count += 1
        assert hashlib.sha256(cache.read_bytes()).hexdigest() == specification['sha256']
digest = hashlib.sha256()
with destination.open('rb') as f:
    for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
        digest.update(block)
summary = dict(host=host, records=count, original_cache_bytes_preserved=True,
               archive=str(destination), bytes=destination.stat().st_size,
               sha256=digest.hexdigest(), device_runs=0)
destination.with_suffix(destination.suffix + '.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary))
