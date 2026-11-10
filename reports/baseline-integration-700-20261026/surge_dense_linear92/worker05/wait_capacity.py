from pathlib import Path
import subprocess,time,json,datetime,concurrent.futures
W=Path(__file__).resolve().parent;ssh='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh';R='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/surge_dense_linear92/worker05';start=time.monotonic()
def status(host):
 p=subprocess.run(['npu-guard','status','--json'] if host=='local' else [ssh,host,'npu-guard status --json'],capture_output=True,text=True,timeout=30)
 return dict(host=host,rc=p.returncode,stdout=p.stdout,stderr=p.stderr)
while time.monotonic()-start<480:
 row=status('openlibing-DevEnv-055409');d=json.loads(row['stdout']);now=datetime.datetime.now(datetime.timezone.utc).isoformat()
 with (W/'CAPACITY_WAIT.jsonl').open('a') as f:f.write(json.dumps(dict(utc=now,**row))+'\n')
 (W/'HEARTBEAT.json').write_text(json.dumps(dict(utc=now,exact_key=['ops-transformer','GDNFwdOp','q35-gva-decode-b512-float16'],phase='waiting for actual idle condition; no new submission',last_output='CAPACITY_WAIT.jsonl',blocker=d.get('reasons'),next='submit once idle snapshot proves changed condition')))
 if d['state']=='idle' and not d.get('reservation',{}).get('locked'):
  with concurrent.futures.ThreadPoolExecutor() as ex:rows=list(ex.map(status,['local','openlibing-DevEnv-661495','openlibing-DevEnv-055409']))
  (W/'attempt07_b512_threehosts.json').write_text(json.dumps(rows,indent=2))
  (W/'attempt07_RETRY_CONDITION.json').write_text(json.dumps(dict(utc=now,changed_condition='Prior long-running holder 894777 has exited; fresh server2 status reports idle and unlocked. Previous guard75 launched no workload.',evidence='CAPACITY_WAIT.jsonl')))
  subprocess.run([ssh,'openlibing-DevEnv-055409',f'python3 {R}/remote_run.py q35-gva-decode-b512-float16 attempt07_b512'],check=True)
  subprocess.run(['python3',str(W/'transport.py'),'get','attempt07_b512'],check=True)
  subprocess.run(['python3',str(W/'collect_result.py'),'attempt07_b512'],check=True)
  break
 time.sleep(25)
else:print('No idle capacity within 480s CPU-side observation window; no unchanged guard retry.')
