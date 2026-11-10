import json,shutil,hashlib,difflib,subprocess,shlex,datetime
from pathlib import Path
W=Path(__file__).resolve().parent;T=W.parent/'transformer_linear';case='q35-gva-decode-b512-bfloat16';owner='baseline700-20261026/gdn_capacity_recurrent1'
ssh='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh';host='openlibing-DevEnv-055409'
def remote(code):
 p=subprocess.run([ssh,host,'python3 -S -B -'],input=code,capture_output=True,text=True,timeout=120)
 if p.returncode:raise RuntimeError(p.stderr)
 return p.stdout
pins=json.loads((W.parent/'gdn_capacity_recheck27/DEPLOYMENT_openlibing-DevEnv-055409.json').read_text())['files']
code='''import json,hashlib\nfrom pathlib import Path\npins=json.loads(%r)\nout={}\nfor p,v in pins.items():\n a=hashlib.sha256(Path(p).read_bytes()).hexdigest();out[p]={'expected':v['expected'],'actual':a,'matches':a==v['expected']}\nprint(json.dumps(out))'''%json.dumps(pins)
x=json.loads(remote(code));(W/'DEPLOYMENT_REVERIFIED.json').write_text(json.dumps(x,indent=2));assert len(x)==164 and all(v['matches'] for v in x.values())
shutil.copytree(T/'recurrent_package',W/'recurrent_package',dirs_exist_ok=True)
for n in ['CASE_RESOURCE_ESTIMATES.json','CONTRACTS.json','RECURRENT_CASES.json']:shutil.copy2(T/n,W/n)
d=W/case;d.mkdir(exist_ok=True)
r=json.loads((W/'recurrent_package/RUNTIME_RECIPE.json').read_text());r['required_environment']['GDN_CASE_OWNER']=owner;r['guard'].update(owner=owner,wait=600)
(W/'recurrent_package/RUNTIME_RECIPE.json').write_text(json.dumps(r,indent=2)+'\n')
old=(T/case/'child.sh').read_text();new=old.replace('/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/transformer_linear',str(W)).replace('baseline700-20261026/transformer_linear',owner);(d/'child.sh').write_text(new)
for line in new.splitlines():
 if line.startswith('export ') and '="'+str(d)+'/' in line:Path(line.split('="')[1].rstrip('"')).mkdir(exist_ok=True)
diff=''
for a,b in [(T/case/'child.sh',d/'child.sh'),(T/'recurrent_package/RUNTIME_RECIPE.json',W/'recurrent_package/RUNTIME_RECIPE.json')]:diff+=''.join(difflib.unified_diff(a.read_text().splitlines(True),b.read_text().splitlines(True),fromfile=str(a),tofile=str(b)))
(W/'adapter.diff').write_text(diff)
(W/'EXACTKEYS.json').write_text(json.dumps([['ops-transformer','GDNFwdOp',case]],indent=2))
(W/'REGISTRATION_CANDIDATE.json').write_text(json.dumps({'exactkey':['ops-transformer','GDNFwdOp',case],'binding':str(W/'recurrent_package/binding.py'),'factory':'build','registered':False,'requires_parent_serial_registration':True},indent=2))
frozen={str(p.relative_to(W)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (W/'recurrent_package').iterdir() if p.is_file()};(W/'FREEZE.json').write_text(json.dumps(frozen,indent=2))
files={str(p):p.read_text() for p in W.rglob('*') if p.is_file() and p.name not in ['SPAWN.json','TASK.txt']}
code='''import json\nfrom pathlib import Path\nfiles=json.loads(%r)\nfor p,s in files.items():\n f=Path(p);f.parent.mkdir(parents=True,exist_ok=True);f.write_text(s)\nd=Path(%r)\nfor n in %r:(d/n).mkdir(exist_ok=True)\nprint('private files installed')'''%(json.dumps(files),str(d),[p.name for p in d.iterdir() if p.is_dir()])
print(remote(code));print('164 deployment pins verified; frozen adapter ready')
