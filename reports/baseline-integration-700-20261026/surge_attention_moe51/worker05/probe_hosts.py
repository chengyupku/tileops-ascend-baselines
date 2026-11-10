import concurrent.futures,datetime,hashlib,json,pathlib,platform,shlex,subprocess,time
W=pathlib.Path(__file__).resolve().parent
SSH='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
probe="""import hashlib,json,pathlib,platform,subprocess,sys
p=pathlib.Path('/opt/tileops-all-pairs/build/sgl-pristine-20261025T2310/source/python/sgl_kernel_npu/sgl_kernel_npu/lib/libsgl_kernel_npu.so')
x={'architecture':platform.machine(),'python':sys.version,'library':str(p),'exists':p.is_file(),'device_initialized':False}
if p.is_file():
 x['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
 r=subprocess.run(['nm','-D',str(p)],capture_output=True,text=True);x['nm_rc']=r.returncode;x['unidex_exports']=[s for s in r.stdout.splitlines() if 'unidex' in s.lower()]
print(json.dumps(x))
"""
def run(host):
 cmd=[sys.executable,'-c',probe] if host=='local' else [SSH,host,'/opt/tileops-all-pairs/bin/allpairs-python -c '+shlex.quote(probe)]
 t=time.monotonic();start=datetime.datetime.now(datetime.timezone.utc).isoformat()
 try:
  r=subprocess.run(cmd,capture_output=True,text=True,timeout=35);return dict(host=host,requested_utc=start,elapsed_seconds=time.monotonic()-t,command=cmd,rc=r.returncode,stdout=r.stdout,stderr=r.stderr)
 except subprocess.TimeoutExpired:return dict(host=host,requested_utc=start,elapsed_seconds=time.monotonic()-t,error='CPU SSH probe timeout')
import sys
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:rows=list(ex.map(run,['local','openlibing-DevEnv-661495','openlibing-DevEnv-055409']))
(W/'HOST_CPU_PREFLIGHT.json').write_text(json.dumps(rows,indent=2)+'\n')
for r in rows:print(r['host'],r.get('rc'),r.get('stdout','').strip(),r.get('error',''))
