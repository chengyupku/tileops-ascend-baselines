"""Foreground, same-guard native process reused only within one pair."""
import os,sys,json,runpy,gc,traceback
from pathlib import Path
root=Path(__file__).resolve().parents[1]
reply=os.fdopen(int(os.environ.pop('BASELINE_NATIVE_REPLY_FD')),'w',buffering=1)
allowed={str(root/'tools'/n) for n in ('baseline_support_math_child_v2.py','baseline_support_math_profile_child_v15.py')}
for line in sys.stdin:
 req=json.loads(line);script=req['argv'][0]
 if script not in allowed:raise ValueError('Unexpected native request')
 saved=[os.dup(1),os.dup(2)];status=0
 with open(req['log'],'a') as log:
  try:
   os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
   os.chdir(req['cwd']);sys.argv=req['argv'];runpy.run_path(script,run_name='__main__')
  except SystemExit as e:status=e.code if isinstance(e.code,int) else (0 if e.code is None else 1)
  except BaseException:
   traceback.print_exc();status=1
  finally:
   gc.collect()
   if 'torch' in sys.modules:
    try:sys.modules['torch'].npu.synchronize()
    except BaseException:traceback.print_exc();status=78
   sys.stdout.flush();sys.stderr.flush()
   for fd,old in zip((1,2),saved):os.dup2(old,fd);os.close(old)
 reply.write(json.dumps({'returncode':status,'pid':os.getpid(),'pgid':os.getpgrp()})+'\n');reply.flush()
reply.close()
