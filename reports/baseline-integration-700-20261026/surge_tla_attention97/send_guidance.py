import pathlib,json,subprocess,os,datetime
P=pathlib.Path(__file__).resolve().parent
jobs=json.loads((P/'CHILD_JOBS.json').read_text())['jobs'];out=[]
for r in jobs:
 n=r['native']; thread=None
 for line in (pathlib.Path(n['directory'])/'events.jsonl').open():
  e=json.loads(line)
  if e.get('type')=='thread.started':thread=e['thread_id'];break
 message='协调指令：97 key已核互斥，保持原scope。请立即写HEARTBEAT（含确切key/下一可执行步骤/CPU预检缺项/复用recipe）。历史相同条件的原同步超时不重放；未跑exact key仍标未知，不能用代表失败判全家失败。优先你scope中可独立执行的非attention key及允许薄接口修复，禁止反复泛查历史。各key完整checker成功立即交RESULT及可机械发布包+CPU真实路由。共同TLA部署缺项由worker01单owner协调；其他worker只提交精确缺项并做独立key准备，禁止重复诊断同一共因。私有组内协调信息在../COORDINATION.json；请确认收到于HEARTBEAT。'
 if r['directory'].endswith('worker01'):message+=' 你是本组共同部署缺项唯一owner，仅当发现实际缺件时补原依赖recipe；已有原实现同步/编译限制不可改kernel/pass。无需额外profiling。'
 env=os.environ.copy();env['CODEX_HOME']=n['home']['path']
 z=subprocess.run(['/root/dyq/agent-bin/codex','queue','--thread',thread,'--message',message],env=env,capture_output=True,text=True,timeout=20)
 out.append({'worker':r['name'],'thread':thread,'returncode':z.returncode,'stdout':z.stdout[-400:],'stderr':z.stderr[-400:]})
(P/'GUIDANCE_RECEIPTS.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
print(json.dumps(out,ensure_ascii=False))
