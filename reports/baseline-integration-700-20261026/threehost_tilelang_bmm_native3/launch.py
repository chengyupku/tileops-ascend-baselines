import pathlib,subprocess,json,time,sys,shlex
P=pathlib.Path(__file__).resolve().parent;W=P.parents[2];ssh=W/'.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh';remote='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_tilelang_bmm_native3'
host,baseline,op,cid=sys.argv[1:5];suffix=sys.argv[5] if len(sys.argv)>5 else '';tag='_'.join([baseline,op,cid])+suffix;key=[baseline,op,cid]
assert key in [[r['baseline'],r['operator'],r['case_id']] for r in json.loads((P/'SCOPE.json').read_text())]
cmd=['npu-guard','run','--owner','baseline700-20261026/threehost_tilelang_bmm_native3','--wait','120','--timeout','180','--','/opt/tileops-all-pairs/bin/allpairs-python',remote+('/run_tla.py' if baseline=='tilelang-ascend' else '/run_case.py'),*key,suffix]
rec=dict(key=key,host=host,argv=cmd,started=time.time());path=P/(tag+'.guard.json');log=P/(tag+'.guard.log');assert not log.exists()
with log.open('x') as f:
 proc=subprocess.Popen([str(ssh),dict(server1='openlibing-DevEnv-661495',server2='openlibing-DevEnv-055409')[host],shlex.join(cmd)],stdout=f,stderr=subprocess.STDOUT)
 while True:
  (P/'HEARTBEAT.json').write_text(json.dumps(dict(phase='guard_wait_or_checker',currentkey=key,host=host,lastprogress=time.time(),blockedreason=None),indent=2))
  try:rc=proc.wait(timeout=25);break
  except subprocess.TimeoutExpired:pass
rec.update(finished=time.time(),returncode=rc);path.write_text(json.dumps(rec,indent=2));print(json.dumps(rec))
(P/'HEARTBEAT.json').write_text(json.dumps(dict(phase='guard_finished',currentkey=key,host=host,lastprogress=time.time(),blockedreason=None if rc==0 else 'See guard receipt'),indent=2))
