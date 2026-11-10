from pathlib import Path
import subprocess,json,datetime,concurrent.futures,tarfile,io,time
W=Path(__file__).resolve().parent
R='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/surge_dense_linear92/worker04'
SSH='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
HOST='openlibing-DevEnv-055409'
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def heartbeat(case,stage):
 (W/'HEARTBEAT.local.json').write_text(json.dumps(dict(utc=now(),exact_key=['ops-transformer','GDNFwdOp',case],phase=stage,last_substantive_output='per-key RESULT and foreground attempt',next='collect original checker')))
def collect(a):
 cmd=f'tar --exclude="*.pt" --exclude="native_logs" --exclude="cache" --exclude="torch_extensions" --exclude="tmp" -cf - -C {R} {a}'
 p=subprocess.run([SSH,HOST,cmd],capture_output=True,check=True)
 with tarfile.open(fileobj=io.BytesIO(p.stdout)) as t:t.extractall(W,filter='data')
def result(case,a):
 A=W/a;g=json.loads((A/'GUARD_RESULT.json').read_text()); c=json.loads((A/'checker.json').read_text()) if (A/'checker.json').exists() else {};lock=json.loads((A/'LOCK_FINAL.json').read_text())
 row=dict(key=['ops-transformer','GDNFwdOp',case],status=c.get('status','not_run_guard_or_process_failure'),reason=c.get('error',c.get('reason')),registered=False,accepted_added=0,checker=str(A/'checker.json') if c else None,guard=str(A/'GUARD_RESULT.json'),native=str(A/'raw.log'),remote_captures=HOST+':'+R+'/'+a,guard_exit=g['returncode'],guard_metadata=lock,input_seed=c.get('input_seed'),timings=dict(requested=lock.get('requested_at'),acquired=lock.get('acquired_at'),end=g['utc'],device_seconds=None,holding_cpu_seconds=None),runtime_ready=False)
 (A/'RESULT.json').write_text(json.dumps(row,indent=2));print(json.dumps(row),flush=True)
 with (W/'RESULT.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
def status(host):
 cmd=['npu-guard','status','--json'] if host=='local' else [SSH,host,'npu-guard status --json']
 p=subprocess.run(cmd,capture_output=True,text=True,timeout=30)
 return dict(host=host,utc=now(),exit=p.returncode,stdout=p.stdout,stderr=p.stderr)
if __name__=='__main__':
 for i,case in enumerate(['q35-gva-decode-b1-float16','q35-gva-decode-b16-float16','q35-gva-decode-b32-float16','q35-gva-decode-b128-float16','q35-gva-decode-b256-float16','q35-397b-tp1-32x8192-float16'],2):
  a=f'attempt{i:02d}'; heartbeat(case,'three host status')
  with concurrent.futures.ThreadPoolExecutor() as e: states=list(e.map(status,['local','openlibing-DevEnv-661495',HOST]))
  (W/(a+'_THREEHOST.json')).write_text(json.dumps(states,indent=2))
  heartbeat(case,'foreground guard')
  with (W/(a+'_transport.log')).open('w') as log:
   p=subprocess.Popen([SSH,HOST,f'cd {R} && /opt/tileops-ci/runtime/bin/python remote_run.py {case} {a}'],stdout=log,stderr=subprocess.STDOUT)
   while True:
    try:rc=p.wait(timeout=30);break
    except subprocess.TimeoutExpired:heartbeat(case,'foreground guard')
  collect(a);result(case,a)
