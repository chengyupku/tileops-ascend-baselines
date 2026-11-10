import json,statistics,time,csv
from pathlib import Path
C=Path(__file__).resolve().parent
inventory={h:json.load(open(C/(h+'.inventory.json'))) for h in ['server1','server2']}
results={h:json.load(open(C/(h+'.results.json'))) for h in inventory}
checks={k:inventory['server1'][k]==inventory['server2'][k] for k in ['hardware','driver_version','sources','adapter_hashes','vendor_hashes','runtime_libs','package_versions']}
plan=json.load(open(C/'PLAN.json'));rows=[]
for task in plan['pairs']:
 entries={h:[r for r in results[h]['results'] if r['id']==task['id']] for h in inventory}
 row={'key':task['key'],'id':task['id'],'complete':all(len(v)==2 and all(r['status']=='valid' for r in v) for v in entries.values())}
 if row['complete']:
  inputs=[r['result']['actual_inputs'] for v in entries.values() for r in v];row['actual_inputs_equal']=all(v==inputs[0] for v in inputs)
  medians={};row['round_mean_us']={};row['round_sample_cv_percent']={}
  for h,items in entries.items():
   samples=[s*1000 for r in items for s in r['result']['timing']['latency_samples_ms']]
   medians[h]=statistics.median(samples)
   row['round_mean_us'][h]=[r['result']['timing']['latency_ms']*1000 for r in items]
   row['round_sample_cv_percent'][h]=[100*r['result']['timing']['latency_stdev_ms']/r['result']['timing']['latency_ms'] for r in items]
  row['pooled_median_us']=medians;row['relative_difference_percent']=100*(medians['server2']/medians['server1']-1)
  row['performance_screen_pass']=abs(row['relative_difference_percent'])<=10 or abs(medians['server2']-medians['server1'])<=1
  row['round_drift_percent']={h:100*(max(v)/min(v)-1) for h,v in row['round_mean_us'].items()}
  row['stable']=all(v<=10 for v in row['round_drift_percent'].values())
  row['pass']=row['actual_inputs_equal'] and row['performance_screen_pass'] and row['stable']
 rows.append(row)
complete=all(r['complete'] for r in rows)
report={'checked_at':time.time(),'static_checks':checks,'rows':rows,'complete':complete,'sampled_equivalence_pass':complete and all(checks.values()) and all(r.get('pass') for r in rows),'scope_limit':'Three NN API/shape cases, two rounds each; not a proof for every baseline library/compiler/deployment. Cold startup and host dispatch are excluded from device-latency equivalence.','shared_host_evidence':'Same boot_id and CPU affinity 0-383; each cgroup quota 48 CPUs, memory limit96GiB. Distinct physical NPUs6 and0.','formal_campaign':'paused, timer disabled; no automatic resume'}
(C/'COMPARISON.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
with (C/'COMPARISON.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['baseline','op','case','complete','server1_median_us','server2_median_us','difference_percent','pass'])
 for r in rows:w.writerow([*r['key'],r['complete'],r.get('pooled_median_us',{}).get('server1'),r.get('pooled_median_us',{}).get('server2'),r.get('relative_difference_percent'),r.get('pass')])
print(json.dumps(report,ensure_ascii=False))
