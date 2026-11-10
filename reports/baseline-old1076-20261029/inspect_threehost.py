import concurrent.futures,json,subprocess,time
from pathlib import Path
p=Path(__file__).resolve().parent
code='''import json,os,subprocess,time
from pathlib import Path
p=Path(ROOT)/'reports/baseline-old1076-20261029'/HOST
l=json.loads((p/'LIVE.json').read_text());ident=l['identity'];proc=Path('/proc')/str(ident['pid']);actual=None
if (proc/'stat').exists():
 s=(proc/'stat').read_text();actual=int(s[s.rfind(')')+2:].split()[19])
v=os.statvfs(p);r=subprocess.run(['npu-guard','status','--json'],capture_output=True,text=True)
print(json.dumps({'at':time.time(),'live':l,'actual_start_ticks':actual,'identity_matches':actual==ident['start_ticks'],'disk_free_bytes':v.f_bavail*v.f_frsize,'npu':json.loads(r.stdout)}))
'''
hosts=[('local','/root/dyq/workspace',None),('server1','/opt/tileops-baseline-cache/workspace','openlibing-DevEnv-661495'),('server2','/opt/tileops-all-pairs/workspace','openlibing-DevEnv-055409')]
def one(args):
 h,root,alias=args;cmd=['python3','-'] if not alias else ['ssh','-T','-o','BatchMode=yes','-o','ClearAllForwardings=yes',alias,'python3 -'];r=subprocess.run(cmd,input=code.replace('ROOT',repr(root)).replace('HOST',repr(h)),text=True,capture_output=True,check=True,timeout=30);return h,json.loads(r.stdout)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:r=dict(ex.map(one,hosts))
(p/'THREEHOST_PHYSICAL.latest.json').write_text(json.dumps(r,indent=2))
print(json.dumps({h:{'matches':v['identity_matches'],'pid':v['live']['identity']['pid'],'phase':v['live']['phase'],'npu':v['npu']['state'],'freeGiB':v['disk_free_bytes']/2**30} for h,v in r.items()}))
