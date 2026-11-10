import pathlib,json,subprocess,datetime,time
P=pathlib.Path(__file__).resolve().parent;W=P.parents[2]
jobs=json.loads((P/'CHILD_JOBS.json').read_text())['jobs'];old=next(j for j in jobs if j['name'].endswith('/worker01'))
assert not any(j['name'].endswith('/worker01r2') for j in jobs)
s=json.loads(subprocess.check_output(['python3','/root/dyq/tools/codex-worker.py','status',old['native']['id']]))
assert s['status']=='completed' and s['exit_code']==0
q=P/'worker01r2';q.mkdir(exist_ok=True)
scope=[x for x in json.loads((P/'worker01/SCOPE.json').read_text()) if x['case_id'] in ['q35-gva-decode-b8-float16','q35-gva-decode-b64-float16','q35-gva-decode-b512-float16'] and x['baseline']=='flash-linear-attention-npu']
assert len(scope)==3
(q/'SCOPE.json').write_text(json.dumps(scope,ensure_ascii=False,indent=2))
contract=(P/'worker01/TASK.txt').read_text().replace('surge_dense_linear92/worker01','surge_dense_linear92/worker01r2')
intro='''本次为协调者纠偏补位，旧worker01已CLI completed且PID身份不再存活，3个exact key独占转移给你。旧worker01错误以代表case失败停止未跑exactkey，本轮用户明确要求实际补测，禁止重复该错误。任务类型=实际补测。先写HEARTBEAT明确接受3key所有权。只跑SCOPE.json里的FLA GDN b8/b64/b512 float16，三个此前均未执行，不是相同失败重试；b256已有真实原失败，不得跑。直接读取兄弟worker03/run.py、entry.py、CPU_PREFLIGHT.json及fla_q35-gva-decode-b4-float16的guard/checker，复用已验证local FLA recipe到你私有路径。先用同一完整原checker跑3项再结论，代表失败不阻止精确补测。不改kernel/compiler/checker/tolerance/math。每项立刻RESULT；若完整成功，交可机械发布登记补丁与CPU真实路由。预检最多3分钟；三机检查仍必需。禁止继续委派。模型gpt-6-astra medium。\n'''
(q/'TASK.txt').write_text(intro+contract)
selfjob=json.loads((P/'SELF_JOB.json').read_text());deadline=datetime.datetime.fromisoformat(json.loads((P/'GROUP.json').read_text())['deadline_utc']).timestamp()
cmd=['python3',str(W/'.cache/all-pairs/full_rollout_20261024/codex_worker_manual_quota.py'),'--native-script','/root/dyq/tools/codex-worker.py','spawn','--cwd',str(W),'--task-file',str(q/'TASK.txt'),'--write',str(q),'--model','gpt-6-astra','--effort','medium','--timeout',str(min(3600,int(deadline-time.time()))),'--notify','queue','--sandbox','danger-full-access','--parent',old['parent_thread'],'--parent-home',selfjob['home']['name'],'--npu-owner','baseline700-20261026/surge_dense_linear92/worker01r2','--npu-timeout','600','--npu-wait','120']
r=subprocess.run(cmd,capture_output=True,text=True);(q/'SPAWN.json').write_text(json.dumps({'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr},indent=2));assert r.returncode==0,r.stderr
n=json.loads(r.stdout);old['active_assignment']=False;jobs.append({'name':'surge_dense_linear92/worker01r2','directory':str(q),'task_file':str(q/'TASK.txt'),'native':n,'count':3,'parent_thread':old['parent_thread'],'active_assignment':True,'replaces':old['native']['id'],'submitted_at':time.time()})
(P/'CHILD_JOBS.json').write_text(json.dumps({'jobs':jobs},indent=2))
a=json.loads((P/'REPLACEMENT_ACTIONS.json').read_text());a.append({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'action':'replaced_incomplete_worker01','old_job':old['native']['id'],'new_job':n['id'],'keys':[x['key'] for x in scope],'reason':'Representative failure was incorrectly used to stop 3 never-run exact keys. Existing worker03 runnable recipe is concrete changed availability. Old CLI exited0 and PID identity gone.'});(P/'REPLACEMENT_ACTIONS.json').write_text(json.dumps(a,ensure_ascii=False,indent=2));print(json.dumps({'id':n['id'],'home':n['home'],'scope_count':3}))
