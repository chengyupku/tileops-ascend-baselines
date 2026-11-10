from control import *
import tarfile,hashlib
D=P/'server1';D.mkdir(exist_ok=False)
with tarfile.open(P/'server1_evidence.tar.gz') as t:t.extractall(D,filter='data')
pins=json.loads((D/'SEALED_HASHES.json').read_text());checked={}
for name,expected in pins.items():
 h=hashlib.sha256()
 with (D/name).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 checked[name]=h.hexdigest()==expected
assert all(checked.values()),checked
(P/'LOCAL_VERIFICATION.json').write_text(json.dumps(dict(verified_files=len(pins),all_hashes_match=True,transport_tar_exit=1,transport_warning=(P/'collection.log').read_text(),explanation='Root directory mtime changed during independent device proof addition; archive decompressed successfully and every presealed evidence file hash matches. No device rerun or evidence alteration.'),indent=2))
hb('evidence_verified_local');print('verified',len(pins))
