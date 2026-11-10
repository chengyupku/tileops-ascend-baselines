from pathlib import Path
import hashlib,json,shutil,time
P=Path(__file__).resolve().parent;W=P.parents[2];S=W/'.cache/all-pairs/remote_native_deepgemm_deepep';sdk=S/'sdk-920b2'
def sha(f):
 with f.open('rb') as h:return hashlib.file_digest(h,'sha256').hexdigest()
expected={};parts=[]
for name in ['assembly_receipt.json','assembly_extend_receipt.json']:
 r=json.loads((sdk/name).read_text());expected.update(r['original_file_sha256']);parts.extend(r['parts'])
checked=[];missing=[]
freeze=json.loads((P/'HOST_FREEZE.json').read_text())['files']
for p,r in freeze.items():
 f=Path(p)
 if not f.is_relative_to(sdk):continue
 rel=str(f.resolve().relative_to(sdk.resolve())) if f.resolve().is_relative_to(sdk.resolve()) else str(f.relative_to(sdk))
 h=expected.get(rel)
 if h:checked.append(dict(path=p,package_relative=rel,sha256=r['sha256'],package_sha256=h,matches=h==r['sha256']))
 else:missing.append(dict(path=p,resolved=str(f.resolve()),relative=rel))
extract=[]
for root in [S/'cann-920b2-extracted',*[Path(x) for x in set(parts)]]:
 f=root/'extraction_receipt.json'
 if f.exists():
  d=P/'provenance'/('sdk_extraction_'+root.name+'.json');shutil.copyfile(f,d);extract.append(dict(path=str(f),sha256=sha(f),receipt=json.loads(f.read_text())))
assert all(x['matches'] for x in checked),[x for x in checked if not x['matches']]
(P/'SDK_AUDIT.json').write_text(json.dumps(dict(at=time.time(),checked=checked,not_in_assembly_manifest=missing,extraction=extract,source='Existing original CANN package extraction and assembly receipts; no header replacement performed'),indent=2));print(json.dumps(dict(checked=len(checked),missing=len(missing),extractions=len(extract))))
