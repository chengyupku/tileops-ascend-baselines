"""Only formally verified same-run variants enter the comparison."""
import pathlib,json,csv,collections
P=pathlib.Path(__file__).resolve().parent
rows=json.loads((P/'RESULTS.json').read_text());tasks=json.loads((P/'TASKS.json').read_text());out=[]
for i,t in enumerate(tasks):
 actual=[r for r in rows if r['index']==i and r['status']=='formal_raw_verified']
 original=next((r for r in actual if r['variant']=='original'),None)
 best=min(actual,key=lambda r:r['us']) if actual else None
 out.append({'index':i,'op':t['op'],'case':t['case'],'formal_variants':{r['variant']:{'us':r['us'],'raw_result_sha256':r['result_sha256'],'guard_sha256':r['guard_sha256'],'aicpu_counts':r['aicpu_counts'],'task_counts':r['task_counts']} for r in actual},'selected_variant':best['variant'] if best else None,'old_eager_us':original['us'] if original else None,'best_us':best['us'] if best else None,'eager_to_best_ratio':original['us']/best['us'] if original and best else None,'status':'same_run_comparable' if original and best else 'pending'})
(P/'COMPARISON.json').write_text(json.dumps(out,indent=2))
with (P/'COMPARISON.csv').open('w') as f:
 cols=['index','op','case','status','selected_variant','old_eager_us','best_us','eager_to_best_ratio'];w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(out)
summary={'total_workloads':len(tasks),'formal_comparable':sum(r['status']=='same_run_comparable' for r in out),'formal_variant_count':sum(len(r['formal_variants']) for r in out),'selected_variants':dict(collections.Counter(r['selected_variant'] for r in out if r['selected_variant'])),'not_published':True,'old_campaign_paused':True}
(P/'COMPARISON_SUMMARY.json').write_text(json.dumps(summary,indent=2));print(summary)
