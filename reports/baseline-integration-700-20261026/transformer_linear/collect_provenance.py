from pathlib import Path
import json,shutil,hashlib
W=Path(__file__).resolve().parent;R=W.parents[2];T=R/'.cache/all-pairs/full_rollout_20261024/remote_ready/server2/group1/resume_20261025/continuation0430';P=T/'tasks/recovery0730-gdn-fixed-abi/chunk_binary';E=W/'binary_provenance';E.mkdir(exist_ok=True)
for n in ['BUILD_PROOF.json','INSTALL_PROOF.json','SOURCE_PINS.json','GUARD_RECEIPT.json']:
 shutil.copyfile(P/n,E/n)
f=T/'ORIGINAL_PACKAGING_METADATA_RESOLUTION.json';shutil.copyfile(f,E/f.name)
files=json.loads((P/'INSTALL_PROOF.json').read_text())['files'];changed=[]
for pin in files:
 f=Path(pin['path'])
 if not f.exists() or hashlib.sha256(f.read_bytes()).hexdigest()!=pin['sha256']:changed.append(pin['path'])
assert not changed
(E/'CURRENT_BINARY_VERIFICATION.json').write_text(json.dumps(dict(checked=len(files),changed=changed),indent=2));print('original installed files verified',len(files))
