from pathlib import Path
import subprocess,json,datetime,concurrent.futures
W=Path(__file__).resolve().parent;R='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/surge_dense_linear92/worker05';ssh='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
for i,case in enumerate(['q35-gva-decode-b8-float16','q35-gva-decode-b64-float16','q35-gva-decode-b512-float16','varlen-v-first-ragged-float16'],2):
 a=f'attempt0{i}_'+case
 def status(host):
  p=subprocess.run(['npu-guard','status','--json'] if host=='local' else [ssh,host,'npu-guard status --json'],capture_output=True,text=True,timeout=40)
  return dict(host=host,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),rc=p.returncode,stdout=p.stdout,stderr=p.stderr)
 with concurrent.futures.ThreadPoolExecutor() as pool:rows=list(pool.map(status,['local','openlibing-DevEnv-661495','openlibing-DevEnv-055409']))
 (W/f'{a}_threehosts.json').write_text(json.dumps(rows,indent=2))
 (W/'HEARTBEAT.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),exact_key=['ops-transformer','GDNFwdOp',case],phase='device submission',last_output=f'{a}_threehosts.json',next='original checker')))
 subprocess.run([ssh,'openlibing-DevEnv-055409',f'python3 {R}/remote_run.py {case} {a}'],check=True)
 subprocess.run(['python3',str(W/'transport.py'),'get',a],check=True)
 subprocess.run(['python3',str(W/'collect_result.py'),a],check=True)
