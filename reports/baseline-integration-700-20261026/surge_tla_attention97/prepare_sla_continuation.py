import pathlib,json,datetime,shutil,subprocess
P=pathlib.Path(__file__).resolve().parent;utc=datetime.datetime.now(datetime.timezone.utc).isoformat();jobs=json.loads((P/'CHILD_JOBS.json').read_text())['jobs'];census=json.loads((P/'ACTIVE_GUARD_CENSUS.json').read_text());assert all(r.get('exit_code')==0 and not r['sample']['guards'] for r in census['hosts'])
keys={'worker01':['serving-b2-float16'],'worker02':['mla-b128-32768-float16'],'worker03':['mla-tp-b6-float16','mla-tp-b64-bfloat16','mla-tp-b64-float16','ds-v2-4k-float16'],'worker05':['llama-8b-b64-p16-float16-float16','mamba2-2.7b-prefill-32k-bfloat16','qwen3-next-prefill-8k-float16'],'worker06':['llama-8b-serving-step-float16-float16']}
for name,cids in keys.items():
 w=P/name;r=next(r for r in jobs if pathlib.Path(r['directory']).name==name);n=r['native'];z=subprocess.run(['python3','/root/dyq/tools/codex-worker.py','status',n['id']],capture_output=True,text=True);s=json.loads(z.stdout);assert s['status']=='completed' and s.get('exit_code')==0
 scope=json.loads((w/'SCOPE.json').read_text());selected=[r['key'] for r in scope if r['case_id'] in cids];assert len(selected)==len(cids)
 archive=w/'SLA_checkpoint_20261027T0527';archive.mkdir(exist_ok=True)
 for f in ['RESULT.json','DELIVERY.json','HEARTBEAT.json','FILEHASHES.json']:
  if (w/f).exists():shutil.copy2(w/f,archive/f)
 (w/'SLA_REMAINING.json').write_text(json.dumps(selected,indent=2))
 clearance={'utc':utc,'previous_job_id':n['id'],'legal_unfinished':True,'ownership_released':True,'device_drained':True,'release_basis':'Original scoped worker completed its final delivery with exit0; group coordinator retains exclusive scope and releases completed job assignment for same-scope continuation. No live child guard on any host.','native_status_snapshot':s,'guard_census':str(P/'ACTIVE_GUARD_CENSUS.json'),'legal_exact_keys':selected,'scope_transfer':False}
 (w/'REPLACEMENT_CLEARANCE.json').write_text(json.dumps(clearance,indent=2))
 task=(w/'TASK.txt').read_text()+'''\n最新控制层纠正（dispatch_root 2026-10-27T05:22:47）：05:31是原交付SLA而非用户硬停；此前要求按05:31停止合法工作被撤销，原目标和SLA时间绝不重置。可以继续已授权、合法未完成的原对接任务，若逾期必须如实记录。禁止恢复已耗尽/HOLD/API不支持或无条件原失败重放。\n本次只续做SLA_REMAINING.json精确key，旧native已exit0完成交付；REPLACEMENT_CLEARANCE.json记录原所有权释放及三机无本组guard。仍先核当前三机状态，不假定设备空闲。复用本目录全部原recipe和CPU准备，已验证的metadata_binding修复可按原条件复用，禁止重复SDK审计、kernel/compiler/pass/checker/seed/tolerance修改。原scope未扩大，完成后保持全scope RESULT合并，旧SLA_checkpoint不覆盖。\n这些key真实未进入有效设备执行（仅队列/准入失败或此前SLA预算不足），这是合法剩余；不得把代表失败替它们下数学结论。每个原guard前台请求 --wait600，原case运行预算<=600，不nested，不后台逃逸。已存在本组旧guard则先等待排空。新尝试唯一时间戳目录并写requested/acquired/child_start/end。只运行一次有条件的真实补测；若原失败明确且无新合法条件则收口，不无限重放。\n05:31时必须给原SLA时点快照，后续继续工作如实标为SLA逾期；不得篡改原deadline。native本次预算按现有调度脚本，不改数据库或worker系统。完毕按现有supervisor通知协议交RESULT/DELIVERY/FILEHASHES/HEARTBEAT，accepted_added=0；成功立即给可机械发布包及原全checker/native/input/guard证据。\n'''
 (w/'TASK_SLA_CONTINUE.txt').write_text(task)
print(json.dumps({'legal_unexecuted':sum(len(v) for v in keys.values()),'workers':list(keys)}))
