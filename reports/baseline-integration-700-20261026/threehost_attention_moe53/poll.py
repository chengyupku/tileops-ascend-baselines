import pathlib,json,subprocess,datetime,sys
P=pathlib.Path(__file__).parent;case=sys.argv[1];tag=sys.argv[2]
h={'phase':'guarded_fullchecker','currentkey':case,'host':'server2','lastprogress':datetime.datetime.now(datetime.timezone.utc).isoformat(),'blockedreason':None};(P/'HEARTBEAT.json').write_text(json.dumps(h))
code="""import pathlib,json,subprocess
O=pathlib.Path('/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_attention_moe53')/CASE
for n in ['stages.jsonl','receipt.json','checker.json','native.json']:
 p=O/n
 if p.exists():
  if n=='stages.jsonl':print(n,p.read_text()[-1800:])
  elif n=='native.json':
   x=json.loads(p.read_text());print(n,str(x.get('calls',x) if isinstance(x,dict) else x)[:2500])
  elif n=='checker.json':
   x=json.loads(p.read_text());print(n,{k:x.get(k) for k in ['status','stage','error','wall_seconds','verification','inputs_preserved']})
  else:print(n,p.read_text())
m=json.loads(pathlib.Path('/run/lock/ascend-npu/device-0.lock').read_text());print('guard',m.get('phase'),m.get('owner'));print(subprocess.run(['ps','-p',str(m.get('child',{}).get('pid')),'-o','etime,pcpu,rss'],capture_output=True,text=True).stdout)
""".replace('CASE',repr(case))
r=subprocess.run(['/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh','openlibing-DevEnv-055409','python3 -'],input=code,text=True,capture_output=True);(P/(tag+'.log')).write_text(r.stdout+r.stderr);print(r.stdout+r.stderr)
