import pathlib,json,sqlite3,fcntl,datetime,os,shutil
D=pathlib.Path(__file__).resolve().parents[1];q=D/'acceptance_root';db=sqlite3.connect('file:/root/dyq/codex-workers/workers.sqlite?mode=ro',uri=True)
with (D/'FLEET_REFILL.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);queue=json.loads((D/'ROOT_CONTROL_QUEUE.json').read_text());entry=next(e for e in queue['tasks'] if e['name']=='acceptance_root');old=entry.get('job_id');row=db.execute('select status,data from jobs where id=?',(old,)).fetchone();assert row[0]=='completed'
 data=json.loads(row[1]);pid=data.get('child_pid');assert not pid or not pathlib.Path(f'/proc/{pid}').exists()
 with (q/'TASK_POLICY.lock').open('a') as policy:
  fcntl.flock(policy,fcntl.LOCK_EX);task=q/'TASK.txt';backup=q/f'TASK.before_successor_{old}.txt';shutil.copy2(task,backup)
  intro='''调度root恢复现有验收职责：前任b6c103c039c448ad已completed，446→456的10项成功保留，不重复审查/跑数值。你沿本TASK既有用户授权的验收和写权限，不新增权限。绝对截止仍2026-10-27T05:31:00Z，CPU-only，禁止NPU。
立即处理新交接：recovery_native5/BMM_EVIDENCE_HANDOFF.json（三项tilelang Bmm历史完整通过：对已有执行关联证据作明确通过/缺口判定，禁止为了登记重复数值）；recovery_native5/GROUPED_HANDOFF.json（FP16继承适配被新worker指出逐组MatMulV3+FP32，须明确当前原实现/API政策结论，关联已登记BF16如有影响由你负责必要审查；dispatch_root不做验收也不改账本）。不要假设所有排队完成通知已消费。向recovery_native5/ACCEPTANCE_DISPOSITION.json写明确处置供执行协调者继续；持有provider/ACCEPTED排他权限仍只你。
还有surge_tla_attention97执行队伍运行中，原生完成通知已永久动态路由至当前acceptance_root，请每分钟检查新PUBLISH_QUEUE/待投递通知，差集处理，不因一批登记结束就提前结束职责。保持接收直到执行队伍没有可提交结果且已交接清楚，或原截止/用户改变任务。纯等待不要算执行agent，不制造审核工作。无需向顶层主会话逐项queue，最终只给总量和具体未解阻塞。

'''
  task.write_text(intro+task.read_text());shutil.copy2(q/'SELF_JOB.json',q/f'SELF_JOB.before_{old}.json')
 entry.setdefault('prior_job_ids',[]).append(old);entry.pop('job_id');entry['ready']=True;entry['parent_self_job']=str(D/'dispatch_root/SELF_JOB.json');entry['changed_condition']='Previous acceptance root completed; new recovery handoff lacks disposition and live execution fleet continues. Restore authorized receiver without rerunning checks or replaying accepted work.'
 tmp=D/'ROOT_CONTROL_QUEUE.json.dispatch.tmp';tmp.write_text(json.dumps(queue,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,D/'ROOT_CONTROL_QUEUE.json')
 print('Requeued existing authorized acceptance role after terminal identity check; original permission list unchanged')
