import json,csv,collections,datetime
from pathlib import Path
p=Path(__file__).resolve().parent
data={h:json.loads((p/(h+'.json')).read_text()) for h in ['local','server1','server2']}
local=data['local']['rows'];remote=[r|{'host':h} for h in ['server1','server2'] for r in data[h]['rows'] if r['status']=='cached' and r['performance_valid'] is True]
rmap=collections.defaultdict(list)
for r in remote:rmap[tuple(r['key'])].append(r)
allkeys={tuple(r['key']) for d in data.values() for r in d['rows']};by=collections.defaultdict(list)
for r in local:by[r['key'][0]].append(r)
rows=[]
for r in local:
 k=tuple(r['key']);rr=rmap[k]
 rows.append({'baseline':k[0],'operator':k[1],'case':k[2],'task_id':r['task_id'],'measured_hostname':r['hostname'],'device_name':r['device_name'],'aicore_count':36,'target_aicore_count':32,'action':'remeasure_on_32core','existing_remote_cache_count':len(rr),'existing_remote_hosts':';'.join(sorted({x['host'] for x in rr})),'source_file':r['source_file'],'source_line':r['source_line'],'cache_key':r['cache_key'],'latency_ms_36core':r['latency_ms'],'correctness_receipt':r['correctness_receipt'],'guard_receipt':r['guard_receipt']})
with (p/'LOCAL_36CORE_RETEST.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(sorted(rows,key=lambda r:(r['baseline'],r['operator'],r['case'])))
summary={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Current formal CACHE.jsonl ledgers found recursively under reports on all three workspaces; no correctness-only records counted as performance caches','hardware':{h:{'hostname':d['hostname'],'aicore':d['hardware']['aicore']['count'],'vectorcore':d['hardware']['vectorcore']['count']} for h,d in data.items()},'counts_by_host':{h:len(d['rows']) for h,d in data.items()},'total_cache_records':sum(len(d['rows']) for d in data.values()),'total_exact_keys':len(allkeys),'local_cache_records':len(local),'local_exact_keys':len({tuple(r['key']) for r in local}),'local_operator_types':len({r['key'][1] for r in local}),'local_workload_cases':len({tuple(r['key'][1:]) for r in local}),'local_with_remote_key':sum(bool(rmap[tuple(r['key'])]) for r in local),'local_only_keys':sum(not rmap[tuple(r['key'])] for r in local),'by_baseline':{b:len(v) for b,v in sorted(by.items())},'local_campaigns':dict(collections.Counter(r['source_file'].split('/reports/')[1].split('/')[0] for r in local)),'errors':{h:d['errors'] for h,d in data.items()},'retest_manifest':str(p/'LOCAL_36CORE_RETEST.csv'),'mutations':'Audit files only. Existing caches retained; no runners, alarms or NPU tests started.'}
(p/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False,indent=2))
