#!/usr/bin/env python3
import pathlib,json,subprocess,shlex
B=pathlib.Path(__file__).resolve().parent
u=json.loads((B/'SOURCE_CLOSURE.json').read_text())['unresolved_quoted_includes']
names=sorted(set(x['include'] for x in u if x['reason']=='external_or_generated'))
code='''import os,pathlib,json,sys
names=json.loads(sys.stdin.readline()); roots=['/usr/local/Ascend/cann-9.1.0','/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_nn11/sources']
found={n:[] for n in names}; special=[]
for root in roots:
 for d,dirs,files in os.walk(root):
  dirs[:]=[n for n in dirs if n not in ['.git','__pycache__']]
  for f in files:
   p=d+'/'+f
   if f=='GenerateEsPackage.cmake': special.append(p)
   for n in names:
    if p.endswith('/'+n): found[n].append(p)
print(json.dumps({'include_filename_candidates':found,'GenerateEsPackage_candidates':special,'limitation':'Filename existence only; no compiler include ordering, ABI, active-branch or generated-output proof. No symlink directory traversal.'}))
'''
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','openlibing-DevEnv-661495','python3 -I -S -B -c '+shlex.quote(code)],input=json.dumps(names)+'\n',text=True,capture_output=True,timeout=60)
assert r.returncode==0,r.stderr
(B/'REMOTE_EXTERNAL_FILES.json').write_text(json.dumps(json.loads(r.stdout),indent=2)+'\n')
print('external filename scan saved')
