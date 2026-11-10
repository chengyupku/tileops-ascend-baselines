import json,pathlib,subprocess
P=pathlib.Path(__file__).resolve().parent
rows=json.load(open(P.parent/'ci-baseline-elementwise-20261101/current-contract-baseline/BASELINE32_PRIVATE.json'))['rows']
rows=[{'key':r['key'],'cache_key':r['cache_key'],'dir':str(pathlib.Path(r['correctness_receipt']).parent),'host':r['host']} for r in rows if r['key'][0] in ('ops-nn','ops-transformer','flash-linear-attention-npu') and r['key'][1] in ('GemmFwdOp','BmmFwdOp','AddFwdOp')]
out=[]
for host,target in [('server1','openlibing-DevEnv-661495'),('server2','openlibing-DevEnv-055409')]:
 selected=[r for r in rows if r['host']==host]
 code='import json,pathlib,hashlib\nrows=json.loads('+repr(json.dumps(selected))+')\nout=[]\n'+'''
for r in rows:
 p=pathlib.Path(r['dir']);items=[]
 for folder in ('compile_cache','compile_debug','runtime','cache'):
  root=p/folder
  if not root.exists():continue
  for q in root.rglob('*.json'):
   try:d=json.load(open(q))
   except:continue
   if not isinstance(d,dict) or 'coreType' not in d:continue
   if 'kernelName' not in d:continue
   items.append({'path':str(q),'coreType':d.get('coreType'),'core_type':d.get('core_type'),'taskRation':d.get('taskRation'),'kernelName':d['kernelName'],'metadata_sha256':hashlib.sha256(q.read_bytes()).hexdigest()})
 out.append({**r,'directory_exists':p.exists(),'metadata':items})
print(json.dumps(out))
'''
 p=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ClearAllForwardings=yes',target,'python3 -'],input=code,text=True,capture_output=True,timeout=240)
 if p.returncode:print(p.stderr);continue
 a=json.loads(p.stdout);out+=a;print(host,'pairs',len(a),'with_metadata',sum(bool(x['metadata']) for x in a),flush=True)
(P/'PAIR_COMPILE_METADATA.json').write_text(json.dumps(out,indent=2))
