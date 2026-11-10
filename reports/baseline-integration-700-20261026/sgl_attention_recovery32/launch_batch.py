import pathlib,json,subprocess,sys,time,shlex,datetime
P=pathlib.Path(__file__).resolve().parent;R='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/sgl_attention_recovery32';SSH='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
label=sys.argv[1];indices=sys.argv[2:];rows=json.loads((P/'SCOPE.json').read_text());p=P/'batches'/label;p.mkdir(parents=True,exist_ok=False)
cmd=['npu-guard','run','--owner','baseline700-20261026/sgl_attention_recovery32','--wait','3600','--timeout','300','--','bash',R+'/baseline-run','sgl-kernel-npu','python',R+'/batch.py',label,*indices]
start=time.time();(P/'HEARTBEAT').write_text(datetime.datetime.now(datetime.timezone.utc).isoformat()+' '+label)
with (p/'guard.log').open('w') as out:r=subprocess.run([SSH,'openlibing-DevEnv-055409',shlex.join(cmd)],stdout=out,stderr=subprocess.STDOUT)
g={'command':cmd,'exit_code':r.returncode,'started_epoch_local':start,'ended_epoch_local':time.time(),'elapsed_s':time.time()-start};(p/'GUARD_RESULT.json').write_text(json.dumps(g,indent=2));print(label,r.returncode,flush=True)
subprocess.run([sys.executable,str(P/'sync.py'),'get'],check=True)
for i in map(int,indices):
 a=P/'attempts'/f'{label}-{i:02d}'
 if not (P/'runs'/a.name).exists():continue
 a.mkdir(exist_ok=False);row=rows[i];(a/'GUARD_RESULT.json').write_text(json.dumps(g|{'key':[row[k] for k in ('baseline','operator','case_id')],'batch_guard_receipt':str((p/'GUARD_RESULT.json').relative_to(P))},indent=2))
subprocess.run([sys.executable,str(P/'refresh.py')],check=True)
