import json,sys,subprocess,datetime,hashlib,concurrent.futures
from pathlib import Path
W=Path(__file__).resolve().parent; T=W.parent/'transformer_linear'
recipe=json.loads((T/'chunk_package/RUNTIME_RECIPE.json').read_text())
pins=json.loads((T/'chunk_package/RUNTIME_PINS.json').read_text())['files']
selected=[p for p in pins if p['path'] in (recipe['opapi'],recipe['nnopbase']) or ('kdapairs_transformer' in p['path'] and p['path'].endswith('.json'))]
files=[str(T/f) for f in ['CONTRACTS.json','CASE_RESOURCE_ESTIMATES.json','run_one.py','chunk_package/binding.py','chunk_package/entry.py','chunk_package/resource_gate.py','chunk_package/RUNTIME_RECIPE.json']]
expected={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files}
cases=[x['case_id'] for x in json.loads((W/'SCOPE.json').read_text())]
payload=r'''
import json,sys,os,hashlib,shutil,datetime
from pathlib import Path
cfg=json.loads(sys.argv[1]); base=Path(cfg['base']); out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':os.uname().nodename}
out['proc_meminfo']=Path('/proc/meminfo').read_text();out['proc_self_cgroup']=Path('/proc/self/cgroup').read_text();out['cgroup_mounts']=[s for s in Path('/proc/self/mountinfo').read_text().splitlines() if ' - cgroup' in s]
out['cgroups']=[]
for root,ln,un in [(Path('/sys/fs/cgroup/memory'),'memory.limit_in_bytes','memory.usage_in_bytes'),(Path('/sys/fs/cgroup'),'memory.max','memory.current')]:
 if (root/ln).exists():
  out['cgroups'].append({'root':str(root),'limit':(root/ln).read_text().strip(),'usage':int((root/un).read_text()),'stat':{k:int(v) for k,v in (s.split() for s in (root/'memory.stat').read_text().splitlines())}})
out['disk']={}
for p in [base,Path('/tmp'),Path('/')]:
 if p.exists():out['disk'][str(p)]=dict(zip(('total','used','free'),shutil.disk_usage(p)))
out['deployment']={}
for p,expected in cfg['files'].items():
 f=Path(p); present=f.is_file(); actual=hashlib.sha256(f.read_bytes()).hexdigest() if present else None
 out['deployment'][p]={'exists':present,'sha256':actual,'expected_sha256':expected,'matches':actual==expected if expected else None}
out['runtime_paths']={p:Path(p).exists() for p in cfg['paths']}
out['cases']={}
for c in cfg['cases']:
 d=base/c;out['cases'][c]={'directory':d.is_dir(),'child':(d/'child.sh').is_file(),'guard_receipt_exists':(d/'GUARD_RECEIPT.json').exists(),'checker_exists':(d/'checker.json').exists()}
 if (d/'checker.json').is_file():
  x=json.loads((d/'checker.json').read_text());out['cases'][c]['checker_summary']={k:v for k,v in x.items() if k in ('status','error','errors','success','passed')}
print(json.dumps(out))
'''
(W/'probe_source.py.txt').write_text(payload)
ssh='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
def collect(host):
 remote=host!='local';base='/opt/tileops-all-pairs/workspace' if remote else '/root/dyq/workspace'
 mapped={f.replace('/root/dyq/workspace',base):h for f,h in expected.items()}
 mapped.update({p['path']:p['sha256'] for p in selected})
 cfg={'base':base+'/reports/baseline-integration-700-20261026/transformer_linear','files':mapped,'cases':cases,'paths':[recipe['opapi'],recipe['nnopbase'],recipe['vendor'],recipe['checker'],*recipe['required_environment'].values()][:4]+[recipe['required_environment'][k] for k in ('ASCEND_HOME_PATH','ASCEND_OPP_PATH')]+['/opt/tileops-ci/runtime/bin/python']}
 import shlex
 cmd=['python3','-S','-B','-c',payload,json.dumps(cfg)]
 if remote:cmd=[ssh,host,shlex.join(cmd)]
 p=subprocess.run(cmd,capture_output=True,text=True,timeout=60)
 result={'returncode':p.returncode,'stderr':p.stderr,'observation':json.loads(p.stdout) if p.returncode==0 else p.stdout}
 (W/('HOST_'+host+'.json')).write_text(json.dumps(result,indent=2));return host,p.returncode
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
 for result in ex.map(collect,['local','openlibing-DevEnv-661495','openlibing-DevEnv-055409']):print(result)
