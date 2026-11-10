import subprocess,sys,json,datetime,shlex,tarfile,io,time
from pathlib import Path
P=Path(__file__).resolve().parent
HOST='openlibing-DevEnv-055409'
OWNER='baseline700-20261026/non_tla_transformer79/fp8_padding_execute8'
a,case=sys.argv[1:]
assert any(r['key'][2]==case for r in json.loads((P/'SCOPE.json').read_text()))
assert a.isalnum()
def heartbeat(phase):
 (P/'HEARTBEAT.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'phase':phase,'case_id':case,'attempt':a}))
cmd=f'cd {shlex.quote(str(P))} && mkdir {a} && FP8_APPROVED_DEVICE_WORKER=1 npu-guard run --owner {OWNER} --wait 600 --timeout 600 -- bash ./ops-transformer-python ./run_fp8_padded.py {a} {shlex.quote(case)} > {a}/guard.log 2>&1; rc=$?; cat /run/lock/ascend-npu/device-0.lock > {a}/GUARD_FINAL.json; echo "$rc" > {a}/exit_code'
(P/(a+'_COMMAND.txt')).write_text('ssh '+HOST+' '+shlex.quote(cmd)+'\n')
heartbeat('guard_submitted')
proc=subprocess.Popen(['ssh',HOST,cmd])
while True:
 try: proc.wait(timeout=55);break
 except subprocess.TimeoutExpired: heartbeat('guard_pending_or_running');print('HEARTBEAT',a,flush=True)
heartbeat('collecting')
data=subprocess.check_output(['ssh',HOST,f'tar -C {P} -cf - {a}'])
tarfile.open(fileobj=io.BytesIO(data)).extractall(P,filter='data')
check=json.loads((P/a/'checker.json').read_text()) if (P/a/'checker.json').exists() else {}
rc=int((P/a/'exit_code').read_text())
route=[]
for f in (P/a/'ASCEND_PROCESS_LOG_PATH').rglob('*.log'):
 for line in f.read_text(errors='replace').splitlines():
  if 'GroupedMatmul' in line and ('bin path:' in line or 'StreamLaunchKernelV2:' in line):route.append(line)
(P/a/'ROUTE.txt').write_text('\n'.join(route))
original=any(str(P/'ops-transformer-vendor') in x and 'bin path:' in x for x in route)
rows=json.loads((P/'RESULT.json').read_text())
for r in rows:
 if r['key'][2]==case:
  r.update(status='verified_pending_publication' if check.get('status')=='verified' and original else 'execution_unverified',evidence=f'{a}/checker.json',guard_exit=rc,original_kernel_route=original,checker_status=check.get('status'),error=check.get('error'))
(P/'RESULT.json').write_text(json.dumps(rows,indent=2));(P/'PUBLISH_CANDIDATES.json').write_text(json.dumps([r for r in rows if r['status']=='verified_pending_publication'],indent=2))
heartbeat('case_finished');print(json.dumps({'case':case,'checker':check.get('status'),'guard_exit':rc,'original_route':original,'error':check.get('error')}),flush=True)
