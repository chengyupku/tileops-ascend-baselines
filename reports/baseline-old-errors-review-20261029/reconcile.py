import json,csv,collections,datetime
from pathlib import Path
root=Path('/root/dyq/workspace'); p=root/'reports/baseline-cache-20261027';out=Path(__file__).resolve().parent
rows=json.load(open(p/'support-classification-20261029/UNION_SNAPSHOT.json'))['rows']
old={r['id']:r for r in rows if r['original_group']=='attempted_noncache1076'}
q=json.load(open(p/'support-run-20261029/QUEUE.json'))['pairs']
manual={r['id']:r for r in json.load(open(p/'support-run-20261029/MANUAL_DECISIONS.json'))}
triage=json.load(open(p/'support-run-20261029/local-TRIAGE.json')); supported={r['id']:r for r in triage['supported']}
records=[]
for id,r in old.items():
 latest=r.get('latest_at_snapshot') or {};error=str(latest.get('error') or latest.get('measurement_status') or latest.get('status'))
 state='已明确分类' if id in manual else '本轮已通过正确性' if id in supported else '已入重测队列' if id in q else '未见本轮接手记录'
 if any(s in error for s in ['guard','GUARD','external_conflict','measured_pending_guard']): action='核对现行guard绑定/测量回执；保留正确性证据，污染性能不得缓存'
 elif any(s in error for s in ['No registered provider','Only tested','before importing','No such file','deployment_blocked','OPP_PATH','composition is retired']): action='恢复历史原版部署及顶层适配；旧注册/主机白名单不能作不支持证据'
 elif 'HCCL' in error:action='检查原始通信环境与拓扑；环境初始化失败不作不支持判定'
 elif any(s in error for s in ['not close','not equal','mismatched','Mismatched']):action='检查输入输出语义与允许的顶层适配，保持checker与原始kernel不变'
 elif any(s in error for s in ['Convert','codegen','layout','Compile']):action='保留原始编译器，检查部署/允许的顶层API或配置，不修改kernel和调度'
 else:action='逐项核对原始API与历史正确性回执，最多三次有变化的顶层适配；不能直接沿用旧错误判不支持'
 records.append({'id':id,'baseline':r['key'][0],'op':r['key'][1],'case':r['key'][2],'state':state,'decision':manual.get(id,{}).get('bucket',''),'host':r['host'],'old_error':error,'measurement':latest.get('measurement',''),'next_action':action})
with (out/'old_errors.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
summary={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'old_error_pairs':len(old),'states':dict(collections.Counter(r['state'] for r in records)),'queue_intersection':len(old.keys()&q.keys()),'manual_intersection':len(old.keys()&manual.keys()),'untaken_by_baseline':dict(collections.Counter(r['baseline'] for r in records if r['state']=='未见本轮接手记录')),'limits':'未见队列记录不等于agent从未阅读。下一步建议为CPU初筛，不是已验证修复或最终不支持结论。未修改运行队列或共享记忆。'}
(out/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(json.dumps(summary,ensure_ascii=False))
