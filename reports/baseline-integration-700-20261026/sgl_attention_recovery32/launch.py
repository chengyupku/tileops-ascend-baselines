import pathlib,json,subprocess,sys,time,shlex,datetime
P=pathlib.Path(__file__).resolve().parent;R='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/sgl_attention_recovery32';SSH='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
label,index=sys.argv[1:];row=json.loads((P/'SCOPE.json').read_text())[int(index)]
p=P/'attempts'/label;p.mkdir(parents=True,exist_ok=False)
cmd=['npu-guard','run','--owner','baseline700-20261026/sgl_attention_recovery32','--wait','3600','--timeout','300','--','python3','-B',R+'/provider_guard.py',label,'bash',R+'/baseline-run','sgl-kernel-npu','python',R+'/run_case.py',row['baseline'],row['operator'],row['case_id'],R+'/runs/'+label]
start=time.time();(P/'HEARTBEAT').write_text(datetime.datetime.now(datetime.timezone.utc).isoformat()+' '+label)
with (p/'guard.log').open('w') as out:r=subprocess.run([SSH,'openlibing-DevEnv-055409',shlex.join(cmd)],stdout=out,stderr=subprocess.STDOUT)
(p/'GUARD_RESULT.json').write_text(json.dumps({'command':cmd,'exit_code':r.returncode,'started_epoch_local':start,'ended_epoch_local':time.time(),'elapsed_s':time.time()-start,'key':[row[k] for k in ('baseline','operator','case_id')]},indent=2));print(label,r.returncode,flush=True)
subprocess.run([sys.executable,str(P/'sync.py'),'get'],check=True)
