from pathlib import Path
# Reuse the exact foreground submission/receipt implementation, without its batch loop.
exec((Path(__file__).resolve().parent/'batch.py').read_text().split('pending=[r for r in rows')[0])
for row in rows:
 if not row.get('attempts') or row['attempts'][-1].get('guard_exit') not in (76,78):continue
 old=row['attempts'][-1]['host'];condition=dict(key=row['key'],prior_host=old,prior_guard=row['attempts'][-1]['receipt'],changed_condition='Different compatible host after current clean status; previous guard stopped workload for lost device visibility or external interference, not numerical checker failure.')
 (P/(row['key'][2]+'_RETRY_CONDITION.json')).write_text(json.dumps(condition,indent=2))
 tried={old}
 while time.time()<END:
  ss=statuses();idle=[s['host'] for s in ss if s.get('data',{}).get('state')=='idle' and s['rc']==0 and s['host'] not in tried]
  if not idle:
   heartbeat(row['key'],'waiting_for_migration_host',condition,'Need healthy different host');time.sleep(20);continue
  h=idle[0];rc=process(row,h,len(row['attempts'])+1)
  if rc==75:continue
  if rc in (76,78):tried.add(h)
  else:break
  if len(tried)==len(HOSTS):break
heartbeat(None,'visibility_migration_finished','Per-key attempts preserved in RESULT.json')
