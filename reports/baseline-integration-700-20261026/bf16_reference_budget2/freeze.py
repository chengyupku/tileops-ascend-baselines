from pathlib import Path
import json,hashlib,time
T=Path(__file__).resolve().parent
sha=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
release=json.loads((T/'PARENT_RELEASE.json').read_text())
pre=json.loads((T/'PRE_RELEASE_PINS.json').read_text());pins=pre['sha256']
shared='/root/dyq/workspace/TileOPs/benchmarks/ascend_all_pairs/ops_transformer/__init__.py'
assert release['approved'] and release['execution_ownership_transferred'] and release['job']=='11eba677f7f24e40'
assert sha(Path(shared))==release['provider_initializer_sha256']
pins[shared]=release['provider_initializer_sha256']
assert all(sha(Path(p))==h for p,h in pins.items()),'pre-release math/native source changed'
pins[pre['deferred_shared_initializer']]=sha(Path(pre['deferred_shared_initializer']))
for f in T.iterdir():
 if f.suffix in ('.py','.sh') or f.name in ('CASES.json','PARENT_RELEASE.json'):pins[str(f.resolve())]=sha(f)
data={'sha256':pins,'release':release,'frozen_at':time.time()}
assert not (T/'FROZEN.json').exists()
(T/'FROZEN.json').write_text(json.dumps(data,indent=2))
for row in json.loads((T/'CASES.json').read_text()):
 d=T/row['key'][2];d.mkdir(exist_ok=True);(d/'FROZEN.json').write_text(json.dumps(data,indent=2))
print('Frozen after release',len(pins))
