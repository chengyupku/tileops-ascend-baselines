import pathlib,json,datetime,collections,os
P=pathlib.Path(__file__).resolve().parent;D=P.parent;f=D/'surge_tla_attention97/REPLACEMENTS.jsonl';events=[]
for line in f.read_text().splitlines():
 x=json.loads(line)
 if x.get('utc','')<'2026-10-27T04:07:00':continue
 old=x.get('previous_status',{});new=json.loads(x['stdout']) if x.get('returncode')==0 else {}
 if old.get('finished_at') and new.get('created_at'):
  events.append({'at':x.get('utc'),'old_job':old.get('id'),'old_status':old.get('status'),'old_finished':old['finished_at'],'new_job':new.get('id'),'new_created':new['created_at'],'vacancy_seconds':new['created_at']-old['finished_at'],'new_native_timeout':new.get('timeout'),'condition':x.get('condition_changed')})
values=[x['vacancy_seconds'] for x in events];o={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source':str(f),'events':events,'old_status_counts':dict(collections.Counter(x['old_status'] for x in events)),'max_recorded_vacancy_seconds':max(values) if values else None,'recorded_over_120_seconds':[x for x in events if x['vacancy_seconds']>120],'notes':'Current process survival is distinct from terminal lifecycle failures. Only completed recorded replacements, not all unfilled target slots, have measurable vacancy here.'};target=P/'NATIVE_REPLACEMENT_CENSUS.json';tmp=P/('NATIVE_REPLACEMENT_CENSUS.json.tmp.'+str(os.getpid()));tmp.write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,target);print(json.dumps({'replacement_events':len(events),'old_statuses':o['old_status_counts'],'max_recorded_vacancy_seconds':o['max_recorded_vacancy_seconds']}))
