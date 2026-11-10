from control import *
cmd=f'npu-guard run --owner baseline700-20261026/threehost_bmm_input_backend1 --wait 120 --timeout 300 -- bash {R}/run.sh > {R}/guard.log 2>&1; rc=$?; echo "$rc" > {R}/guard.exit; cat /run/lock/ascend-npu/device-0.lock > {R}/GUARD_FINAL.json; exit "$rc"'
(P/'SUBMISSION.json').write_text(json.dumps(dict(argv=[SSH,HOST,cmd],time=time.time()),indent=2))
with (P/'transport.log').open('w') as log:
 p=subprocess.Popen([SSH,HOST,cmd],stdout=log,stderr=subprocess.STDOUT)
 while True:
  hb('single_capture_guard_pending_or_running')
  try:rc=p.wait(timeout=25);break
  except subprocess.TimeoutExpired:pass
(P/'transport.exit').write_text(str(rc));hb('capture_guard_finished');print('exit',rc)
print(remote(f'cat {R}/guard.exit {R}/GUARD_FINAL.json; tail -20 {R}/guard.log'))
