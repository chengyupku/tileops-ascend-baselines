import datetime, fcntl, hashlib, json, os, pathlib, time
ROOT=pathlib.Path('/root/dyq/workspace');B=ROOT/'reports/baseline-cache-20261027';D=B/'inspection-20261029T001544'
read=lambda p:json.loads(p.read_text())
def save(p,x):
    tmp=p.with_name(p.name+'.inspection-001544.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,p)
hosts=['local','server1','server2'];now=time.time();utc=datetime.datetime.fromtimestamp(now,datetime.timezone.utc).isoformat()
before={h:read(D/(h+'.json'))['summary'] for h in hosts};after={h:read(D/(h+'-after.json'))['summary'] for h in hosts}
effects={h:read(D/(h+'-EFFECT.json')) for h in hosts};changes=read(D/'ASSIGNMENT_CHANGES.json')
audits={}
for h in hosts:
    paths=[D/(h+'-AUDIT.json'),D/(h+'-AUDIT-delta.json')];a=[read(p) for p in paths]
    audits[h]={'paths':list(map(str,paths)),'checked':sum(x['checked'] for x in a),'usable':sum(x['usable'] for x in a),'rejected':sum(x['rejected'] for x in a),'fresh_hashes':all(x['fresh_hashes'] for x in a),'device_runs':0}
assert all(a['rejected']==0 for a in audits.values())
assert all(s['identities']['supervisor']['matches'] for s in after.values())
new_sha=hashlib.sha256((B/'PLAN.json').read_bytes()).hexdigest()
assert all(read(D/(h+'-PUBLISHED.json'))['plan_sha256']==new_sha for h in hosts)
assert all(not effects[h]['previous_111_held_device_attempts'] for h in hosts)
next_check='2026-10-29T04:15:44.214085+00:00'
record={
    'kind':'four_hour_three_host_cache_inspection_and_pending_rebalance',
    'due_at_utc':'2026-10-29T00:15:44.214085+00:00','checked_at':now,'checked_at_utc':utc,
    'report':str(D/'RESULT.json'),'scope_count':6419,'assignment_revision':7,'plan_sha256':new_sha,
    'before':before,'after':after,'audits':audits,'effects':effects,
    'audit_usable':sum(a['usable'] for a in audits.values()),'audit_rejected':0,
    'accepted_last_hour_at_initial_inventory':sum(s['windows']['1']['cached'] for s in before.values()),
    'accepted_last_four_hours_at_initial_inventory':sum(s['windows']['4']['cached'] for s in before.values()),
    'ranked_problems':[
        {'priority':1,'issue':'Ready work concentrated on server2 after local/server1 normal sweep completion',
         'initial_ready':{'local':0,'server1':0,'server2':70},'idle_cards':2,
         'observed_idle_minutes_at_inventory':{h:(before[h]['at']-before[h]['current']['at'])/60 for h in ['local','server1']},
         'confidence':'high; current PLAN/journal/RETRY/holds plus exited native identities and free CPU supervisor locks',
         'action':'At natural pair boundary redistribute only68 genuinely ready/unattempted tasks.42 SGL tasks move; vLLM stays server2. Original scripts resume with identical PLAN revision7 on all hosts.',
         'remaining_at_handoff':changes['ready_after'],'estimated_work_seconds':changes['estimated_work_seconds'],
         'expected_accepted_rate_gain':None,'risk_cost':'No runtime/provider changes. Source stopped before publication; exclusive CPU locks and STOP tokens held; previous caches immutable.',
         'evidence':['ASSIGNMENT_CHANGES.json','local-MIGRATION_PREFLIGHT.json','server1-MIGRATION_PREFLIGHT.json','server2-MIGRATION_PREFLIGHT.json']},
        {'priority':2,'issue':'Long-tail cases have higher compilation/reference/runtime cost; recent server2 mean attempt ~314s',
         'evidence':'server2.json windows1; native child running and no resource_wait',
         'action':'Balance by coarse expected time, not raw total historical assignments. Preserve full checker/profiler coverage and original timeouts. No intrinsic algorithm or compiler changes.',
         'expected_accepted_rate_gain':None,'confidence':'observed cost high; per-case forecasts approximate'},
        {'priority':3,'issue':'Original deployment/contract/numerical/compiler failures remain unavailable for caching',
         'largest_recent_local_group':'27 original guard-admission failures,1636 attempt-wall seconds; source/version constraints require coherent deployment, not bare unchanged retry',
         'action':'Preserve failure history and precise holds. No failed-pair retries, no pin relaxation, no blanket success. Current rebalance is only ready work.',
         'evidence':['local-analysis.json','server1-analysis.json','server2-analysis.json'],'confidence':'high for logged causes; unexecuted holds remain inferred availability restrictions'}],
    'operational_findings':[
        'Initial local/server1 exited identities and older LIVE correspond to sweep_complete with ready0, not lost workers. Their host guards were unlocked.',
        'No resource_wait journal wall time in current1h/4h windows; no evidence of spurious hardware-lock queuing.',
        'All three restored CPU supervisors/foreground guards have matching native identities; new exact cached outputs observed.',
        'Remote clocks approximately93 seconds behind local. Use host-local time for rolling windows; tar future-mtime notices reflect this known skew, not source changes.',
        'Two utilization samples five seconds apart are insufficient for sustained utilization estimates.',
        'Server1 disk approximately4.46GiB free; no source rebuild or foreign file deletion performed.',
    ],
    'verification':{
        'previous111_hold_skips':{h:effects[h]['previous_111_hold_consumption'] for h in hosts},
        'previous111_held_device_attempts':0,
        'previous356_hold_skips':effects['local']['previous_356_hold_consumption'],
        'previous356_held_device_attempts':len(effects['local']['previous_356_held_device_attempts']),
        'post_rebalance_events':{h:effects[h]['post_launch_counts'] for h in hosts},
        'incoming_task_outcomes':{h:effects[h]['incoming_outcomes'] for h in hosts},
        'all_initial_cache_prefixes_unchanged':True,
        'note':'Fresh delta audits include later rows than effect snapshots. Actual new caches prove deployment works; short sample does not establish sustained accepted-rate improvement.'},
    'remaining_at_post_snapshot':{h:{k:after[h][k] for k in ['cached','ready','held','tested_noncache_unheld']} for h in hosts},
    'unchanged':['6419 exactcase scope','all successful cached ownership','baseline kernel/compiler/schedule/checker','guard and timing protocol','historical failures and holds'],
    'changed':['PLAN ownership for42 unattempted SGL pairs','CAMPAIGN assignment revision/counts','normal-boundary CPU supervisors restart'],
    'retry_changes':None,'new_holds':0,
    'followup':[
        'Continue three existing supervisors on PLAN7. At next inspection count current ownership, not historical journal launch host.',
        'If ready0 and sweep_complete, treat as normal queue completion, do not resurrect unchanged failures or holds.',
        'Successful cache count is distinct from tested/temporarily-held/failed scope; full universal success not claimed.',
        'Retain all previous precise holds and fix only actual changed environment/route conditions with RETRY revision.',
        'Keep original4h idle-only alarm anchor; next04:15UTC, no periodic interrupting message or fleet.'
    ],'next_check':next_check}
assert not (D/'RESULT.json').exists();save(D/'RESULT.json',record)
entry={k:record[k] for k in ['kind','due_at_utc','checked_at','checked_at_utc','report','scope_count','assignment_revision','audit_usable','audit_rejected','accepted_last_hour_at_initial_inventory','accepted_last_four_hours_at_initial_inventory','verification','next_check']}
entry.update(audits=audits,rebalanced_ready=changes['ready_after'])
with (B/'INSPECTIONS.lock').open('a+') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    with (B/'INSPECTIONS.jsonl').open('a') as f:f.write(json.dumps(entry,ensure_ascii=False)+'\n')
    save(B/'LAST_INSPECTION.json',entry)
message='''Baseline cache 每4小时巡检。仅在会话空闲时单次唤醒，启动后自动解除；禁止打断活跃修复。保持原四小时锚点，下次2026-10-29 04:15UTC，不重开旧fleet或agents。用户已授权三张卡并行。
权威PLAN revision7，总6419 pair，成功缓存归属不变。当前总归属local3877/server1601/server21941。00:15巡检发现local/server1可执行队列正常跑完，server2尚有70项；在当前pair正常完成后，把剩余68项按预计耗时分为local21/server121/server226，42个未运行SGL迁往空闲卡。vLLM保留server2部署。三端PLAN哈希一致，CPU脚本已恢复，并有迁移项实际完整验证/profiler缓存通过。证据inspection-20261029T001544/RESULT.json与ASSIGNMENT_CHANGES.json。上轮只是sweep_complete，不是worker死亡；今后ready0且sweep_complete不重启无变化失败项。
三端：local=/root/dyq/workspace；server1=ssh openlibing-DevEnv-661495，独立缓存工作区/opt/tileops-baseline-cache/workspace，逻辑0映射/dev/davinci6；server2=ssh openlibing-DevEnv-055409，/opt/tileops-all-pairs/workspace，/dev/davinci0。禁止改一号原算子开发目录。远端无systemd，原tools/baseline_cache_service.sh监督CPU进程。实际当前分工看PLAN/CAMPAIGN/LIVE，不能把脚本active或重新分配当产出。
先告知检查计划，按需读任务记忆baseline_cache_20261027。统一时间窗批量检查三端LIVE/journal/CACHE/QUARANTINE/RETRY、CPU监督/guard/前台child PID+start+boot、实际缓存增长、近1h/4h吞吐、竞争/排队/利用率及磁盘。统计按当前PLAN host/id过滤，迁出历史不可重复计数。远端时钟较本地慢约93秒，窗口用各host时钟。
使用tools/audit_baseline_cache.py --cache <host>/CACHE.jsonl --fresh-hashes --output <新时间戳文件>审计完整checker/全部输出/输入保护/原case合约、3warmup/15profiler/1GiB L2flush、无Event、独占guard回执、原生源码/运行库哈希。审计历史回执不是重新执行，缓存只同case/host/软件环境复用。有效缓存绝不重跑；失败只有真实条件变化且记录RETRY revision才重试。无人为5分钟空窗，不终止外部任务、不绕过guard、不因排队收回任务。
先完成批量盘点并按共性、范围、实际损失、可修复性、证据排序，告知最重要计划后亲自修复部署/环境/调度或精确暂挂。禁止修改原baseline kernel/compiler/schedule/checker或解除源码pin。QUARANTINE修改持QUARANTINE.lock；监督脚本只合并本pair新增自动暂挂。仅pair正常结束边界切换脚本与归属，验证真实CACHE增长及清单消费。不得从单case失败扩大到全库。
保留既有精确证据：FP16 GDN接口31、远端缺provider85、server1 DeepGEMM47缺tilelang.ascend；16:15新增local356接口/准入、FFT10缺FFTS、BF16 27原冻结源码不符；20:15新增local35 ops-transformer FP16 GDN接口只收BF16、server1 42/server2 34 SGL FP16 GDN原cumsum输出FP32却storeFP16。推断暂挂不是执行失败或成功。旧356项已全部直接跳过且0次设备重跑。111组local35/server142全部跳过，server2暂17项其余尚未走到，均0重跑。PTO guard身份错误来自旧registration固定历史owner，不是硬件锁排队；禁止伪造owner或改pin。原kernel/编译器/数值检查均未改。
实际检查后记录checked_at、三端审计、动作与前后效果至INSPECTIONS.jsonl/LAST_INSPECTION.json；持RECURRING_ALARMS.lock将对应节点标记checked_at/inspection_report。投递不等于巡检。只更新自己任务baseline_cache_20261027，不覆盖其他分支。持续每4h直到用户停止。'''
with (B/'RECURRING_ALARMS.lock').open('a+') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX);p=B/'RECURRING_ALARMS.json';a=read(p)
    save(D/'RECURRING_ALARMS.before.json',a)
    target=[m for m in a['milestones'] if abs(m['due_at']-1793232944.2140849)<.01];assert len(target)==1
    target[0].update(checked_at=now,inspection_report=str(D/'RESULT.json'))
    assert any(abs(m['due_at']-1793247344.2140849)<.01 for m in a['milestones'])
    a.update(message=message,assignment_revision=7,pending_at_revision7=changes['ready_after']);save(p,a)
state_path=ROOT/'.agent_memory/THREADS/baseline-inspector-20261023/STATE.json'
with state_path.with_name('.state.lock').open('a+') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX);state=read(state_path);m=state['baseline_cache_20261027'];save(D/'MEMORY_before_baseline_cache.json',m)
    m.update(updated_at=utc,status='three_host_cache_sweep_running_after_tail_rebalance',assignment_revision=7,local=3877,server1_assigned=601,server2=1941,
        last_inspection=entry,recurring_next_due_at=next_check,
        current_ready={h:after[h]['ready'] for h in hosts},current_cached={h:after[h]['cached'] for h in hosts},current_held={h:after[h]['held'] for h in hosts},
        current_counts_observed_at={h:after[h]['host_time'] for h in hosts})
    m['tail_rebalance_revision7']={'report':str(D/'RESULT.json'),'assignment_changes':str(D/'ASSIGNMENT_CHANGES.json'),'moved':42,'handoff_ready':changes['ready_after'],'estimated_work_seconds':changes['estimated_work_seconds'],'all_cached_ownership_preserved':True,'no_environment_provider_changes':True,'effect':record['verification']}
    m['active_followup']='All3 original CPU scripts restored on PLAN7;68 ready split21/21/26 at handoff. Actual moved-pair caches passed fresh audits. Next04:15UTC idle-only. If ready0+sweep_complete, do not relaunch completed/unchanged failed work. Held/failed scope is not success.'
    m['next_actions']=['Current PLAN7-filtered actual cache/ready counts and native process/guard identity on all3 hosts','Fresh audit without device reruns; preserve exact holds/failure history; check source and disk','Continue original4h idle-only inspections. Queue completion is distinct from every pair successful.']
    save(state_path,state)
print(json.dumps({k:entry[k] for k in ['checked_at_utc','assignment_revision','audit_usable','audit_rejected','accepted_last_hour_at_initial_inventory','accepted_last_four_hours_at_initial_inventory','rebalanced_ready','next_check']},ensure_ascii=False,indent=2))
