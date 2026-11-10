import subprocess,json,concurrent.futures
from pathlib import Path
C=Path(__file__).resolve().parent
hosts=[('server1','openlibing-DevEnv-661495','/opt/tileops-baseline-cache/workspace'),('server2','openlibing-DevEnv-055409','/opt/tileops-all-pairs/workspace')]
def collect(item):
 h,a,r=item
 script="""import pathlib,json
h=pathlib.Path(ROOT)/'reports/baseline-remote-equivalence-full-20261030'/HOST
s=json.loads((h/'STATE.json').read_text())
r=json.loads((h/'RESULTS.json').read_text()) if (h/'RESULTS.json').exists() else []
pid=s['identity']['pid']
try:
 fields=pathlib.Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split();s['process_alive']=int(fields[19])==s['identity']['start_ticks']
except FileNotFoundError:s['process_alive']=False
print(json.dumps({'state':s,'results':r}))
""".replace('ROOT',repr(r)).replace('HOST',repr(h))
 p=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ClearAllForwardings=yes',a,'python3 -'],input=script,text=True,capture_output=True,check=True)
 x=json.loads(p.stdout);(C/(h+'.results.json')).write_text(json.dumps(x,ensure_ascii=False,indent=2));return h,x['state'],[(v['key'],v['round'],v['status'],v['result'].get('error')) for v in x['results']]
with concurrent.futures.ThreadPoolExecutor(2) as pool:
 for x in pool.map(collect,hosts):print(json.dumps(x,ensure_ascii=False))
