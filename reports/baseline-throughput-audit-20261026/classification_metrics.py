"""Classify delivered labels conservatively; does not execute any baseline."""
import json,pathlib,datetime,csv,collections
P=pathlib.Path(__file__).resolve().parent
start=datetime.datetime(2026,10,26,8,tzinfo=datetime.timezone.utc).timestamp();end=start+3600
old={tuple(r[k] for k in ('baseline','operator','case_id')):r for r in csv.DictReader(open(P.parent/'baseline-status-20261026/PRIORITY1751_0800.csv'))}
rows=[];jobrows=[]
for h in ('local','server1','server2'):
 x=json.load(open(P/(h+'_classification.json')));offset=x['local_started']-x['host_now']
 for j in x['jobs']:
  if not j['write'] or '/tasks/' not in j['write'][0] and '/production/' not in j['write'][0]:continue
  finish=j.get('finished_at');finish=finish+offset if finish else None
  if not finish or not(start<=finish<=end):continue
  jr={'host':h,'id':j['id'],'task':j['write'][0].split('/')[-1],'status':j['status'],'created_at':j['created_at']+offset,'finished_at':finish,'elapsed_minutes':(finish-j['created_at']-offset)/60,'result':j['result']}
  choices=[a for a in j['artifacts'] if pathlib.Path(a['path']).name in ('EXACT_DISPOSITION.json','EXACT_KEY_DISPOSITIONS.json','RESULT.json')]
  if choices:
   a=next((a for a in choices if 'DISPOSITION' in a['path']),choices[0]);d=a['data'];rr=d if isinstance(d,list) else d.get('rows',d.get('results',d.get('dispositions',[])))
   if isinstance(rr,dict):rr=list(rr.values())
   for r in rr:
    if not isinstance(r,dict) or not isinstance(r.get('key'),list) or len(r['key'])!=3:continue
    key=tuple(r['key']);label=r.get('classification') or r.get('status') or r.get('disposition','unknown');low=label.lower()
    category='undecided_or_blocked'
    if any(v in low for v in ('excluded','non_target','reasoned_exclusion','contract_exclusion','schedule_required_exclusion')):category='no'
    elif ('verified' in low and 'unverified' not in low) or any(v in low for v in ('source_contract_resolved','source_api_eligible','original_whole_api_source_supported')):category='yes_or_source_eligible'
    prior=old.get(key,{});prior_closed=prior.get('joined_status') in ('excluded_non_target','original_passed','original_verified','verified')
    rows.append({'host':h,'job_id':j['id'],'key':list(key),'label':label,'category':category,'in_priority1751':key in old,'previous_joined_status':prior.get('joined_status'),'previous_owner_status':prior.get('owner_status'),'previously_closed_0804':prior_closed,'finished_at':finish,'evidence':a['path']})
  jr['row_count']=sum(r['job_id']==j['id'] for r in rows);jobrows.append(jr)
latest={}
for r in sorted(rows,key=lambda r:r['finished_at']):latest[tuple(r['key'])]=r
uniq=list(latest.values());new=[r for r in uniq if not r['previously_closed_0804'] and r['in_priority1751']]
summary={'window_start_utc':'2026-10-26T08:00:00Z','window_end_utc':'2026-10-26T09:00:00Z','counts':{'terminal_jobs':len(jobrows),'job_statuses':dict(collections.Counter(j['status'] for j in jobrows)),'delivered_rows':len(rows),'unique_pairs':len(uniq),'within1751':sum(r['in_priority1751'] for r in uniq),'latest_label_categories':dict(collections.Counter(r['category'] for r in uniq)),'new_vs_0804_categories':dict(collections.Counter(r['category'] for r in new)),'previously_closed_0804':sum(r['previously_closed_0804'] for r in uniq)},
 'caveats':['Completed row deliveries are not all definitive decisions. Failed deployments/holds remain undecided unless source eligibility is explicit.', 'New-vs-0804 uses the nearest available joined snapshot, 4m48s after window start; later repeated interim classifications are not proven first-time work.', 'Source-eligible includes records whose runtime is blocked. It is not a numerical pass count.', 'Labels are reported evidence classifications, not an independent re-audit of baseline semantics.'], 'jobs':jobrows,'rows':uniq}
(P/'CLASSIFICATION_METRICS.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print(json.dumps(summary['counts'],ensure_ascii=False,indent=2))
print('byhost',dict(collections.Counter(r['host'] for r in uniq)))
with (P/'CLASSIFICATION_ROWS.csv').open('w') as f:
 fields=list(uniq[0]);w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(uniq)
