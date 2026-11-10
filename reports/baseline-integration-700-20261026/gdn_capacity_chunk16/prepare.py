import json,hashlib,subprocess,shlex
from pathlib import Path
W=Path(__file__).resolve().parent
ssh='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
cfg={'scope':json.loads((W/'SCOPE.json').read_text()),'deployment':json.loads((W.parent/'gdn_capacity_recheck27/DEPLOYMENT_openlibing-DevEnv-055409.json').read_text())}
payload=r'''
import json,hashlib,sys,shutil,difflib,datetime
from pathlib import Path
cfg=json.load(sys.stdin)
W=Path('/root/dyq/workspace/reports/baseline-integration-700-20261026/gdn_capacity_chunk16');W.mkdir(parents=True,exist_ok=True)
T=Path('/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/transformer_linear')
def write(p,d):p.write_text(json.dumps(d,indent=2))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
checks={p:{'expected':v['expected'],'actual':sha(Path(p))} for p,v in cfg['deployment']['files'].items()}
write(W/'DEPLOYMENT_VERIFIED.json',checks)
assert len(checks)==164 and all(v['expected']==v['actual'] for v in checks.values())
write(W/'SCOPE.json',cfg['scope'])
P=W/'chunk_package';P.mkdir(exist_ok=True)
for name in ['binding.py','direct_acl.py','entry.py','resource_gate.py','RESOURCE_ESTIMATES.json','RUNTIME_PINS.json','RUNTIME_RECIPE.json']:
 shutil.copyfile(T/'chunk_package'/name,P/name)
for name in ['CASE_RESOURCE_ESTIMATES.json','CONTRACTS.json','CHUNK_CASES.json']:
 shutil.copyfile(T/name,W/name)
owner='baseline700-20261026/gdn_capacity_chunk16'
f=P/'RUNTIME_RECIPE.json';r=json.loads(f.read_text());r['required_environment']['GDN_CASE_OWNER']=owner;r['guard'].update(owner=owner,wait=600);write(f,r)
diffs=[]
diffs.extend(difflib.unified_diff((T/'chunk_package/RUNTIME_RECIPE.json').read_text().splitlines(True),f.read_text().splitlines(True),fromfile=str(T/'chunk_package/RUNTIME_RECIPE.json'),tofile=str(f)))
contracts=json.loads((W/'CONTRACTS.json').read_text());est=json.loads((W/'CASE_RESOURCE_ESTIMATES.json').read_text())
keys=[]
for s in cfg['scope']:
 c=s['case_id'];d=W/c;d.mkdir(exist_ok=True)
 assert s['status']=='blocked' and c in contracts and c in json.loads((W/'CHUNK_CASES.json').read_text())
 old=T/c/'checker.json'
 if old.exists():assert json.loads(old.read_text()).get('status') not in ('verified','measured')
 child=(T/c/'child.sh').read_text();new=child.replace(str(T),str(W)).replace('baseline700-20261026/transformer_linear',owner)
 (d/'child.sh').write_text(new)
 diffs.extend(difflib.unified_diff(child.splitlines(True),new.splitlines(True),fromfile=str(T/c/'child.sh'),tofile=str(d/'child.sh')))
 for line in new.splitlines():
  if line.startswith('export ') and '="'+str(d)+'/' in line:
   Path(line.split('="',1)[1].rstrip('"')).mkdir(parents=True,exist_ok=True)
 keys.append({'exactkey':[s['baseline'],s['operator'],c],'params':contracts[c]['params'],'tensors':contracts[c]['tensors'],'required_envelope':est[c],'prior_checker_exists':old.exists(),'prior_guard_receipt_exists':(T/c/'GUARD_RECEIPT.json').exists()})
write(W/'EXACTKEYS.json',keys)
write(W/'REGISTRATION_CANDIDATES.json',{'status':'frozen_pending_fullcheck_and_parent_review','registered':False,'accepted_added':0,'adapter':str(P/'binding.py'),'factory':'binding.build','exactkeys':[x['exactkey'] for x in keys]})
(W/'REHOME.diff').write_text(''.join(diffs))
write(W/'FROZEN_ADAPTER.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'deployment_pins_verified':len(checks),'files':{str(p):sha(p) for p in list(P.iterdir())+[W/'CONTRACTS.json',W/'CASE_RESOURCE_ESTIMATES.json',W/'CHUNK_CASES.json'] if p.is_file()},'unchanged_code':{n:sha(P/n)==sha(T/'chunk_package'/n) for n in ['binding.py','direct_acl.py','entry.py','resource_gate.py']},'checker':r['checker'],'runtime_pins_count':len(json.loads((P/'RUNTIME_PINS.json').read_text())['files'])})
print(json.dumps({'verified':len(checks),'cases':len(keys),'package':str(P)}))
'''
(W/'prepare_remote.py').write_text(payload)
p=subprocess.run([ssh,'openlibing-DevEnv-055409',shlex.join(['python3','-S','-B','-c',payload])],input=json.dumps(cfg),text=True,capture_output=True)
(W/'prepare.stdout.log').write_text(p.stdout);(W/'prepare.stderr.log').write_text(p.stderr)
print(p.returncode,p.stdout,p.stderr[:2000]);raise SystemExit(p.returncode)
