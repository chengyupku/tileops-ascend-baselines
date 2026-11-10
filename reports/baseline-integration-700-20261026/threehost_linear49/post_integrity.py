from pathlib import Path
import hashlib,json,datetime,subprocess
P=Path(__file__).resolve().parent
remote=str(P).replace('/root/dyq/workspace','/opt/tileops-all-pairs/workspace') if str(P).startswith('/root/') else str(P)
local=str(P).startswith('/root/')
names=['LOCAL_DEPENDENCY_FREEZE.json'] if local else ['MIGRATED_NATIVE_FREEZE.json','RECURRENT_NATIVE_FREEZE.json']
results=[]
for n in names:
 data=json.loads((P/n).read_text());bad=[]
 for x in data['files']:
  p=Path(x['path']);sha=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
  if sha!=x['sha256']:bad.append({'path':str(p),'expected':x['sha256'],'actual':sha})
 results.append({'manifest':n,'files':len(data['files']),'mismatches':bad})
if local:
 for n in ['SNAPSHOT_HASHES.json','COMPLETE_ORIGINAL_SOURCE_HASHES.json']:
  data=json.loads((P/n).read_text());bad=[q for q,h in data.items() if hashlib.sha256((P/q).read_bytes()).hexdigest()!=h]
  results.append({'manifest':n,'files':len(data),'mismatches':bad})
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':'local' if local else 'server1','checks':results,'passed':all(not x['mismatches'] for x in results),'device_access':False}
(P/'POST_RUN_INTEGRITY.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
r=subprocess.run(['stat','-f','-c','%T %d %i',str(P)],text=True,capture_output=True)
(P/'MOUNT_PROVENANCE.json').write_text(json.dumps({'command':['stat','-f','-c','%T %d %i',str(P)],'stdout':r.stdout,'returncode':r.returncode,'purpose':'retain filesystem backing evidence for raw stat device versus proc maps device'},indent=2))
