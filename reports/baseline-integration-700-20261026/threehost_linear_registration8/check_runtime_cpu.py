import pathlib,json,hashlib,subprocess,shlex,datetime
O=pathlib.Path(__file__).resolve().parent;pins=json.loads((O/'RUNTIME_PINS.json').read_text());result={}
script='''import sys,json,pathlib,hashlib,platform
pins=json.load(sys.stdin);bad=[]
for p,h in pins.items():
 try:actual=hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
 except OSError as e:bad.append([p,type(e).__name__]);continue
 if actual!=h:bad.append([p,"hash_mismatch"])
print(json.dumps(dict(host=platform.node(),arch=platform.machine(),count=len(pins),mismatches=bad)))
'''
for arch in ('x86_64','aarch64'):
 union={}
 for item in pins.values():
  if item['host']['architecture']!=arch:continue
  for p,h in item['runtime'].items():
   if p in union:assert union[p]==h,p
   union[p]=h
 if arch=='x86_64':cmd=['python3','-B','-c',script]
 else:cmd=['/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh','openlibing-DevEnv-661495','python3 -B -c '+shlex.quote(script)]
 proc=subprocess.run(cmd,input=json.dumps(union),text=True,capture_output=True,timeout=180)
 if proc.returncode:result[arch]={'error':proc.stderr[-1200:],'returncode':proc.returncode}
 else:result[arch]=json.loads(proc.stdout)
result['utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();result['passed']=all(result[k].get('mismatches')==[] for k in ('x86_64','aarch64'))
(O/'RUNTIME_CPU_CHECK.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:({'count':v.get('count'),'mismatch_count':len(v.get('mismatches',[])),'error':v.get('error')} if isinstance(v,dict) else v) for k,v in result.items()}))
