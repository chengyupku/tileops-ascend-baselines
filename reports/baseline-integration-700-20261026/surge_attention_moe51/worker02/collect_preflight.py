import pathlib,subprocess,concurrent.futures,json,time,datetime
p=pathlib.Path(__file__).parent; code=(p/'preflight.py').read_text();ssh='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
def run(host):
 cmd=['python3','-'] if host=='local' else [ssh,host,'python3 -']; t=time.time();r=subprocess.run(cmd,input=code,text=True,capture_output=True,timeout=60)
 (p/(host+'-preflight.json')).write_text(r.stdout);(p/(host+'-receipt.json')).write_text(json.dumps({'command':cmd,'utc_end':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed':time.time()-t,'rc':r.returncode,'stderr':r.stderr}))
 print(host,r.returncode,r.stderr[:100])
with concurrent.futures.ThreadPoolExecutor(3) as ex:list(ex.map(run,['local','openlibing-DevEnv-661495','openlibing-DevEnv-055409']))
