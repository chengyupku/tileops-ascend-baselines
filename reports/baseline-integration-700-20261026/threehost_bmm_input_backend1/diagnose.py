from control import *
hb('pure_cpu_diagnostic')
assert (P/'transport.exit').read_text()=='1'
review=json.loads((P/'CPU_LDD.json').read_text());assert review['rc']==0 and not any(x in review['stdout'].lower() for x in ['libascend','torch_npu','not found'])
upload('cpu_diagnostic.py')
cmd=f'env -u LD_LIBRARY_PATH TORCH_DEVICE_BACKEND_AUTOLOAD=0 FLA_NPU_DISABLE_PTH=1 DNNL_VERBOSE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 timeout --signal=TERM --kill-after=5 120 /opt/tileops-ci/runtime/bin/python -I -S -B {R}/cpu_diagnostic.py > {R}/cpu_backend.log 2>&1; rc=$?; echo "$rc" > {R}/cpu_backend.exit; exit "$rc"'
(P/'CPU_SUBMISSION.json').write_text(json.dumps(dict(argv=[SSH,HOST,cmd]),indent=2))
with (P/'cpu_transport.log').open('w') as f:
 p=subprocess.Popen([SSH,HOST,cmd],stdout=f,stderr=subprocess.STDOUT)
 while True:
  hb('pure_cpu_diagnostic_running')
  try:rc=p.wait(timeout=25);break
  except subprocess.TimeoutExpired:pass
print('cpu exit',rc);print(remote(f'tail -15 {R}/cpu_backend.log'));hb('pure_cpu_diagnostic_finished')
