#!/usr/bin/env python3
"""Run ON the fixed target only after separate private-write authorization.
Stages original sources into a NEW private directory; never builds/loads libraries.
Upload this bundle separately after authorization; no runtime/OPP transfer.
"""
import argparse, hashlib, json, pathlib, platform, shutil, socket
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--authorized-private-write',action='store_true',required=True)
p.add_argument('--destination',required=True)
a=p.parse_args()
assert platform.machine()=='aarch64', 'Wrong architecture'
assert socket.gethostname()=='a5d0b50dc68f', 'Target container identity changed; revalidate fixed SSH host openlibing-DevEnv-661495'
b=pathlib.Path(__file__).resolve().parent
m=json.loads((b/'BUNDLE_MANIFEST.json').read_text())
d=pathlib.Path(a.destination)
parent=pathlib.Path('/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026')
assert d.is_absolute() and d.parent.resolve()==parent.resolve()
assert d.name.startswith('quant_deploy_8c35d3ec5cd545ed_'), 'Use a fresh task-private directory'
assert not d.exists(), 'Refuse existing destination'
items=[]
for f in m['files']:
 rel=pathlib.PurePosixPath(f['path']); assert not rel.is_absolute() and '..' not in rel.parts
 src=(b/'source_delta'/rel) if f['bundled'] else pathlib.Path(m['remote_base'])/rel
 assert src.is_file() and hashlib.sha256(src.read_bytes()).hexdigest()==f['sha256'], str(src)
 items.append((src,rel,f['sha256']))
# All preflight checks before any write. Partial destination on I/O failure is retained for diagnosis.
d.mkdir()
for src,rel,h in items:
 out=d/rel;out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,out)
 assert hashlib.sha256(out.read_bytes()).hexdigest()==h
(d/'SOURCE_STAGE_RECEIPT.json').write_text(json.dumps({'files':len(items),'build_executed':False,'closure_proven':False,'manifest_sha256':hashlib.sha256((b/'BUNDLE_MANIFEST.json').read_bytes()).hexdigest()},indent=2)+'\n')
print('Source staging complete. Build blocked: see DEPENDENCY_MANIFEST.json. No installation or runtime validation performed.')
