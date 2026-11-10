"""Reuse original native child runtime across check/profile, never across guards."""
import os,sys,json,subprocess,time
from pathlib import Path
_original_run=subprocess.run
class PairNativePool:
 def __init__(self):self.proc=None;self.reader=None;self.signature=None;self.records=[]
 def run(self,command,*args,**kwargs):
  targets=('baseline_support_math_child_v2.py','baseline_support_math_profile_child_v15.py')
  script=next((str(x) for x in command if str(x).endswith(targets)),None) if isinstance(command,list) else None
  if not script:return _original_run(command,*args,**kwargs)
  env=kwargs.get('env',dict(os.environ));sig=tuple(env.get(k,'') for k in ('ASCEND_CUSTOM_OPP_PATH','LD_LIBRARY_PATH','ASCEND_HOME_PATH','OPS_MATH_GUARD_OWNER'))
  if self.proc is None:
   rd,wr=os.pipe();worker=Path(__file__).with_name('baseline_pair_native_worker_v15.py');e=dict(env);e['BASELINE_NATIVE_REPLY_FD']=str(wr)
   try:self.proc=subprocess.Popen([sys.executable,'-B',str(worker)],stdin=subprocess.PIPE,stdout=kwargs['stdout'],stderr=subprocess.STDOUT,env=e,pass_fds=(wr,),text=True)
   finally:os.close(wr)
   self.reader=os.fdopen(rd);self.signature=sig
  if sig!=self.signature:raise RuntimeError('Native runtime environment differs; refuse reuse')
  if self.proc.poll() is not None:raise RuntimeError('Native worker exited unexpectedly')
  idx=command.index(script);request={'argv':command[idx:],'cwd':str(kwargs.get('cwd',os.getcwd())),'log':str(Path(kwargs['stdout'].name).resolve())}
  start=time.perf_counter();self.proc.stdin.write(json.dumps(request)+'\n');self.proc.stdin.flush();line=self.reader.readline()
  if not line:raise RuntimeError('Native worker response missing')
  result=json.loads(line)
  if result['pid']!=self.proc.pid or result['pgid']!=os.getpgrp():raise RuntimeError('Native worker identity/group mismatch')
  self.records.append({'script':script,'seconds':time.perf_counter()-start,**result})
  return subprocess.CompletedProcess(command,result['returncode'])
 def close(self):
  if self.proc:
   self.proc.stdin.close();self.proc.wait(timeout=30);self.reader.close()
   if self.proc.returncode:raise RuntimeError('Native worker did not exit cleanly')
pool=PairNativePool()
def install():subprocess.run=pool.run
