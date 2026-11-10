from pathlib import Path
import json,hashlib,datetime
P=Path(__file__).resolve().parent;r=json.loads((P/'ORIGINAL92_RECIPE.json').read_text());pins=json.loads((P/'ORIGINAL_SOURCE_PINS.json').read_text());paths={x['path']:x['sha256'] for x in pins};paths.update({r['opapi']:None,r['nnopbase']:None,'/usr/local/Ascend/cann-9.1.0/aarch64-linux/ascend_toolkit_install.info':None})
rows=[]
for n,old in paths.items():
 p=Path(n)
 if p.is_file():
  s=p.stat();sha=hashlib.sha256(p.read_bytes()).hexdigest();rows.append({'path':n,'sha256':sha,'size':s.st_size,'inode':s.st_ino,'device':s.st_dev,'prior_sha256':old,'prior_match':old is None or sha==old})
 else:rows.append({'path':n,'missing':True})
(P/'NATIVE_FREEZE.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':rows},indent=2));print(len(rows),sum(x.get('missing',False) for x in rows),sum(not x.get('prior_match',True) for x in rows))
