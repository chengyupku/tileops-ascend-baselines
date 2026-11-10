import pathlib,json,csv,hashlib,datetime,collections,sqlite3,os
R=pathlib.Path(__file__).resolve().parent
scope=json.load(open(R/'SCOPE608.json'));keys={tuple(x['key']) for x in scope};accepted=json.load(open(R/'PARENT_ACCEPTED.json'));by={tuple(x['key']):x for x in accepted};assert len(by)==len(accepted)==608 and set(by)==keys,('coverage',len(by),len(keys-set(by)))
orig={tuple(x['key']):x for x in json.load(open(R/'SUBMITTED_DECISIONS.json'))};rows=[];corrections=[]
for k in sorted(keys):
 a=by[k]
 if 'audited_row' in a:x=dict(a['audited_row'])
 else:
  x=dict(orig[k]);assert hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True).encode()).hexdigest()==a['row_sha256'],k
 if a.get('reason_override'):x['reason']=a['reason_override']
 assert x['decision']==a['decision'] and x['decision'] in ('要接','不要接') and x['reason'] and x['evidence']
 x.update(baseline=k[0],operator=k[1],case_id=k[2],reviewed_at=a.get('rechecked_at',a['reviewed_at']),review_note=a['note']);rows.append(x)
 if k in orig and orig[k]['decision']!=x['decision']:corrections.append({'key':k,'before':orig[k]['decision'],'after':x['decision'],'reason':x['reason']})
def csvwrite(path,rr,fields):
 with path.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader()
  for r in rr:w.writerow({k:(json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v) for k,v in r.items() if k in fields})
fields=['baseline','operator','case_id','decision','reason','evidence','reviewed_at']
csvwrite(R/'DECISIONS608.csv',rows,fields);(R/'DECISIONS608.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows));(R/'CORRECTIONS.json').write_text(json.dumps(corrections,ensure_ascii=False,indent=2))
sample_report=json.load(open(R.parent/'baseline-sample-recheck-20261026/RESULT.json'));samples=[dict(x,baseline=x['key'][0],operator=x['key'][1],case_id=x['key'][2],reviewed_at=sample_report['classification_finished_at']) for x in sample_report['results']];allrows=rows+samples;assert len({(x['baseline'],x['operator'],x['case_id']) for x in allrows})==611
csvwrite(R/'ALL611_DECISIONS.csv',allrows,fields)
old=json.load(open(R.parent/'baseline-reconciled-20261026/SUMMARY.json'));pending=list(csv.DictReader(open(R.parent/'baseline-reconciled-20261026/DEFINITE_PENDING.csv',encoding='utf-8-sig')));newyes=[dict(x,status='definite_pending') for x in allrows if x['decision']=='要接'];pending=[dict(x,decision='要接') for x in pending]+newyes;assert len({(x['baseline'],x['operator'],x['case_id']) for x in pending})==len(pending)
csvwrite(R/'ALL_DEFINITE_PENDING.csv',pending,['baseline','operator','case_id','decision','status','reason','evidence'])
plan=json.load(open(R/'PLAN.json'));start=datetime.datetime.fromisoformat(plan['started_at']);now=datetime.datetime.now(datetime.timezone.utc);elapsed=(now-start).total_seconds();counts=collections.Counter(x['decision'] for x in rows);counts611=collections.Counter(x['decision'] for x in allrows)
from observe_all import once
health=once();assert health['active']==0 and health['failed']==0,('jobs not all successful terminal',health['active'],health['failed'])
classifiers=[w for w in health['workers'] if w['name'].startswith('worker')];cstart=min(w['created_at'] for w in classifiers);cend=max(w['finished_at'] for w in classifiers);cusage=sum(w['wall_s'] for w in classifiers)/(8*(cend-cstart));totalusage=sum(w['wall_s'] for w in health['workers'])/(8*elapsed)
rawjobs=json.load(open(R/'JOBS.json'))['jobs'];launchdelays=[];db=sqlite3.connect('file:/root/dyq/codex-workers/workers.sqlite?mode=ro',uri=True)
for w in health['workers']:
 j=json.loads(db.execute('select data from jobs where id=?',(w['id'],)).fetchone()[0]);assert j['effort']=='medium';launchdelays.append((int(j['child_stamp'].split(':')[-1])-int(j['supervisor_stamp'].split(':')[-1]))/os.sysconf('SC_CLK_TCK'))
db.close()
ratios=[]
for hn,hf in [('HISTORY.jsonl','live_survival'),('ALL_HISTORY.jsonl','active_survival')]:
 for line in (R/hn).read_text().splitlines():
  hv=json.loads(line)
  if hv.get(hf) is not None:ratios.append(hv[hf])
summary=dict(active_liveness_samples=len(ratios),minimum_observed_active_survival=min(ratios),started_at=start.isoformat(),completed_at=now.isoformat(),deadline=plan['deadline'],within_one_hour=elapsed<=3600,elapsed_seconds=round(elapsed,1),scope=608,number_correction='611-3=608，用户提到609；已在启动时纠正，未重复计样本。',decisions=dict(counts),all611=dict(counts611),unclassified=0,accepted_pairs_per_hour=round(608*3600/elapsed,1),firstpass_submitted=592,firstpass_evidence_gaps=16,firstpass_binary_corrections=len(corrections),jobs=health['jobs'],classifier_jobs=8,review_jobs=health['jobs']-8,all_workers_medium=True,completed_jobs=health['completed'],failed_jobs=health['failed'],job_completion_rate=health['completed']/health['jobs'],peak_parallel_workers=8,classifier_slot_utilization=round(cusage,4),whole_task_worker_slot_utilization=round(totalusage,4),native_child_launch_delay_s={'min':min(launchdelays),'max':max(launchdelays)},dispatch_return_s={'min':min(g['returned_at']-g['submitted_at'] for g in rawjobs),'max':max(g['returned_at']-g['submitted_at'] for g in rawjobs)},api_queue_time=None,api_queue_note='native事件未提供服务端排队时间，不能以spawn返回或日志静默冒充排队。notification=queued仅为完成消息入队，不是agent等待执行。',npu_executions=0,scope_note='全部为CPU源码分类；要接不等于已对接或已数值通过。',global_observed_ledger={'observed_pairs':10833,'completed_including_user_retained':6078,'original_verified':2459,'user_retained_without_reaudit':3619,'definite_pending':len(pending),'excluded':old['excluded']+counts611['不要接'],'unclassified':0},worker_detail=str(R/'ALL_LIVE.json'),decisions_file=str(R/'DECISIONS608.csv'))
assert sum([6078,len(pending),summary['global_observed_ledger']['excluded']])==10833
(R/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));csvwrite(R/'AGENT_TIMINGS.csv',health['workers'],['name','id','effort','assigned','submitted','status','wall_s','commands','failed_commands'])
report=['本轮608 pair分类完成（另含先前三样本的611总表）。',json.dumps(summary,ensure_ascii=False,indent=2),'\nAgent逐项行为与耗时（assigned/submitted为各自任务范围；复核不得重复计吞吐）：']
for w in health['workers']:report.append(f"{w['name']} | {w['id']} | {w['wall_s']/60:.2f}分钟 | scope {w['assigned']} / binary {w['submitted']} | commands {w['commands']} | {w['status']} | {w['last_message']}")
report.append('\n低效原因：原分类提示词虽已修正默认要接规则，但部分agent仍漏查必需flags/全状态输出，或把内部tile参数、外部算法分解当适配。已按共性集中复核并更正；未观察到本批进程失活、NPU排队或转去开发。服务端模型排队无观测字段，不伪造零排队。利用率定义为worker任务活跃秒数/(8槽×观测时长)，不是CPU或模型实际计算利用率。')
(R/'REPORT.txt').write_text('\n'.join(report));print(json.dumps(summary,ensure_ascii=False,indent=2))
