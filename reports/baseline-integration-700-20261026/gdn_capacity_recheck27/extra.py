import json,hashlib,subprocess,shlex,concurrent.futures
from pathlib import Path
W=Path(__file__).resolve().parent;T=W.parent/'transformer_linear';files={}
for package in ['chunk_package','recurrent_package']:
 r=json.loads((T/package/'RUNTIME_RECIPE.json').read_text());pins=json.loads((T/package/'RUNTIME_PINS.json').read_text())['files']
 for p in pins:
  if p['path']==r['opapi'] or (p['path'].startswith(r['vendor']+'/') and p['path'].endswith(('.o','.json'))):files[p['path']]=p['sha256']
 for name in ['binding.py','direct_acl.py','entry.py','RUNTIME_RECIPE.json']:
  p=T/package/name;files[str(p).replace('/root/dyq/workspace','/opt/tileops-all-pairs/workspace')]=hashlib.sha256(p.read_bytes()).hexdigest()
code="""import json,hashlib,sys,datetime,shutil
from pathlib import Path
files=json.loads(sys.argv[1]);out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':{}}
for p,h in files.items():
 f=Path(p);a=hashlib.sha256(f.read_bytes()).hexdigest() if f.is_file() else None;out['files'][p]={'exists':f.is_file(),'matches':a==h,'sha256':a,'expected':h}
p=Path('/opt/tileops-all-pairs/workspace');out['workspace_disk']=dict(zip(('total','used','free'),shutil.disk_usage(p))) if p.exists() else None
print(json.dumps(out))"""
ssh='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
def run(host):
 cmd=['python3','-S','-B','-c',code,json.dumps(files)]
 if host!='local':cmd=[ssh,host,shlex.join(cmd)]
 p=subprocess.run(cmd,capture_output=True,text=True,timeout=60);x=json.loads(p.stdout) if p.returncode==0 else {'error':p.stderr};(W/('DEPLOYMENT_'+host+'.json')).write_text(json.dumps(x,indent=2));print(host,p.returncode,sum(v['matches'] for v in x.get('files',{}).values()),len(files))
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:list(ex.map(run,['local','openlibing-DevEnv-661495','openlibing-DevEnv-055409']))
