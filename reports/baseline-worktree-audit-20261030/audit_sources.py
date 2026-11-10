import subprocess,json,pathlib
base=pathlib.Path('/root/dyq/workspace/ascend_baselines'); refs={}
for p in base.iterdir():
 if (p/'.git').exists():
  raw=subprocess.check_output(['git','-C',str(p),'ls-tree','-r','HEAD']).decode(); refs[p.name]={l.split('\t',1)[1]:l.split()[2] for l in raw.splitlines() if l.startswith('100')}
code='''import os,json,hashlib
refs=json.loads(input()); out=[]
roots=['/opt/tileops-all-pairs/workspace/ascend_baselines','/opt/tileops-baseline-cache/workspace/ascend_baselines']
for root in roots:
 if os.path.isdir(root):
  for name in os.listdir(root):
   p=os.path.join(root,name)
   if name in refs and os.path.isdir(p): out.append((name,p))
for task in ['arithmetic9_20261027','logroots4_20261028','lerp_exp4_20261028']:
 out.append(('ops-nn','/opt/tileops-all-pairs/workspace/.cache/'+task+'/ops-nn/source'))
for name,p in out:
 if not os.path.isdir(p):continue
 diff=[];missing=[];checked=0
 for f,h in refs[name].items():
  q=os.path.join(p,f)
  if not os.path.isfile(q): missing.append(f);continue
  b=open(q,'rb').read(); actual=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\\0'+b).hexdigest();checked+=1
  if actual!=h:diff.append(f)
 print(json.dumps(dict(path=p,checked=checked,changed=diff,missing_count=len(missing))))
'''
import shlex
for host in ['openlibing-DevEnv-661495','openlibing-DevEnv-055409']:
 r=subprocess.run(['ssh',host,'python3 -c '+shlex.quote(code)],input=json.dumps(refs),text=True,capture_output=True)
 pathlib.Path('reports/baseline-worktree-audit-20261030/'+host+'.jsonl').write_text(r.stdout)
 print(host,r.returncode,r.stdout,r.stderr)
