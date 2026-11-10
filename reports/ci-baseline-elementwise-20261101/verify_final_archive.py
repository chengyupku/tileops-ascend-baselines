"""CPU streaming verification of every archived evidence file; no extraction."""
import hashlib
import json
from pathlib import Path
import tarfile

BASE = Path(__file__).resolve().parent
actual = {}
expected = None
with tarfile.open(BASE/'ELEMENTWISE32_FINAL_EVIDENCE.tar.gz', 'r|gz') as archive:
    for member in archive:
        assert member.isfile(), member.name
        stream = archive.extractfile(member)
        if member.name == 'ARCHIVE_MANIFEST.json':
            assert expected is None
            expected = json.load(stream)
        else:
            assert member.name not in actual
            h = hashlib.sha256()
            n = 0
            for block in iter(lambda:stream.read(1024*1024), b''):
                h.update(block)
                n += len(block)
            assert n == member.size
            actual[member.name] = dict(bytes=n,sha256=h.hexdigest())
            if len(actual) % 100000 == 0:
                print(json.dumps(dict(verified_files=len(actual))),flush=True)
        archive.members.clear()
assert expected is not None and actual == expected['files']
assert len(actual) == 890140
assert expected['source_revision'] == '963229b72e148227639b4400e454a4184a85a956'
assert expected['snapshot_sha256'] == '810b5db2425d37b38e0aca728b5076e45ed848d48687f56c289a32d30c33520f'
result = dict(status='passed',files=len(actual),all_file_bytes_and_sha256_verified=True,
              source=expected['source_revision'],snapshot_sha256=expected['snapshot_sha256'])
(BASE/'FINAL_ARCHIVE_MANIFEST_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
