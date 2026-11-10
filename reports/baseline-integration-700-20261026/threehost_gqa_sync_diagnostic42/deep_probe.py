import subprocess,json,pathlib,os,hashlib
p=pathlib.Path('/usr/local/Ascend/ascend-toolkit/latest').resolve();env=os.environ.copy();env['LD_LIBRARY_PATH']=str(p/('aarch64-linux' if os.uname().machine=='aarch64' else 'x86_64-linux')/'lib64')+':'+str(p/'tools/msdebug/lib')+':'+env.get('LD_LIBRARY_PATH','')
out=[]
def run(a):
 try:
  r=subprocess.run(a,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=25);out.append({'argv':a,'rc':r.returncode,'output':r.stdout})
 except Exception as e:out.append({'argv':a,'error':str(e)})
run([str(p/'tools/msdebug/bin/msdebug'),'--no-lldbinit','--batch','-o','help ascend info cores','-o','help ascend info tasks','-o','settings list target','-o','settings list plugin.process'])
run(['bash','-c','ls -l /proc/debug_switch /dev/dbg* 2>&1; cat /proc/debug_switch 2>&1; id; cat /proc/sys/kernel/yama/ptrace_scope 2>&1'])
for a in [['--help'],['op','--help']]:run([str(p/'tools/profiler/bin/msprof')]+a)
run(['bash','-c',f"find {p}/tools -maxdepth 4 -type f \\( -name '*prof*' -o -name '*trace*' \\) | head -90"])
run(['bash','-c',f"rg -n -i 'sdk.?trace|pc.?sampl|aic.*pc|dump.*reg' {p}/tools/ascend_system_advisor/asys --glob '*.py' | head -90"])
w=pathlib.Path('/opt/tileops-all-pairs/workspace');src=w/'ascend_baselines/tilelang-ascend/examples/gqa_fwd_varlen/gqa_fwd_varlen.py'
out.append({'source':str(src),'sha256':hashlib.sha256(src.read_bytes()).hexdigest() if src.exists() else None})
print(json.dumps(out,indent=2))
