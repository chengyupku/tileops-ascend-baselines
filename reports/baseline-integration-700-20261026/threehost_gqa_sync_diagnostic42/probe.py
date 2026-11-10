import subprocess,json,platform,pathlib,os,datetime
out={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'machine':platform.machine(),'host':platform.node(),'commands':[]}
def run(cmd):
 try:
  p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=25);out['commands'].append({'argv':cmd,'rc':p.returncode,'output':p.stdout})
 except Exception as e:out['commands'].append({'argv':cmd,'error':str(e)})
run(['bash','-c','readlink -f /usr/local/Ascend/ascend-toolkit/latest; readlink -f /usr/local/Ascend/cann; cat /usr/local/Ascend/driver/version.info'])
root=pathlib.Path('/usr/local/Ascend')
for pattern in ['**/tools/msdebug/bin/msdebug','**/tools/bisheng_compiler/bin/llvm-objdump','**/tools/bisheng_compiler/bin/bisheng','**/tools/profiler/bin/msprof']:
 seen=set()
 for f in root.glob(pattern):
  f=f.resolve()
  if str(f) in seen:continue
  seen.add(str(f));run([str(f),'--version' if f.name in ['bisheng','llvm-objdump'] else '--help'])
run(['bash','-c',"find /usr/local/Ascend -maxdepth 6 -type f \\( -iname '*debug*' -o -iname '*trace*' \\) | head -100"])
run(['npu-guard','status','--json'])
print(json.dumps(out,indent=2))
