import pathlib,subprocess,json,shlex,datetime,time
P=pathlib.Path(__file__).resolve().parent
R='/opt/tileops-all-pairs/workspace/'+str(P.relative_to('/root/dyq/workspace'))
SSH='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
H='openlibing-DevEnv-055409'
files=['SCOPE.json','run_case.py','baseline-run','native_log_capture.py','integration700_sgl_gdn_remote38.py','route_install.py','CHECKER_PIN.json','guard_one.py']
data=subprocess.run(['tar','-C',str(P),'-cf','-',*files],capture_output=True,check=True).stdout
subprocess.run([SSH,H,'mkdir -p '+shlex.quote(R)+' && tar -C '+shlex.quote(R)+' -xf -'],input=data,check=True)
cid='q35-397b-tp1-32x8192-bfloat16'
cmd=[SSH,H,'python3 '+shlex.quote(R+'/guard_one.py')+' '+shlex.quote(cid)]
(P/'SUBMISSION.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':cmd},indent=2))
with (P/'submit.log').open('w') as f:
 proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT)
 while proc.poll() is None:
  (P/'HEARTBEAT.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'key':['sgl-kernel-npu','GDNFwdOp',cid],'owner':'baseline700-20261026/surge_dense_linear92/worker07r2','phase':'foreground_guard_pending_or_running','last_output':'three host preflight and private wrappers deployed','next':'collect exact full checker receipt','ssh_pid':proc.pid},indent=2))
  try:proc.wait(timeout=45)
  except subprocess.TimeoutExpired:pass
artifact='runs/exact_r2_01--GDNFwdOp--'+cid
data=subprocess.run([SSH,H,'tar -C '+shlex.quote(R)+' -cf - '+shlex.quote(artifact)],capture_output=True,check=True).stdout
subprocess.run(['tar','-C',str(P),'-xf','-'],input=data,check=True)
print((P/artifact/'GUARD_RESULT.json').read_text(),flush=True)
