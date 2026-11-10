"""Fail closed until all 66 selected attempts have local raw evidence audit."""
import pathlib,json,csv,statistics,collections
P=pathlib.Path(__file__).resolve().parent
status=json.loads((P/'STATUS.json').read_text());selected=json.loads((P/'SELECTION.json').read_text());audit=json.loads((P/'ARCHIVE_AUDIT.json').read_text());verified={r['key']:r for r in audit};measurements={(r['ordinal'],r['version']):r for r in status['rows']};rows=[]
for item in selected:
 i=item['ordinal'];v=item['version'];r=dict(measurements[(i,v)]);key=f'runs/{i:02d}/{v}'
 assert key in verified and r['measurement_sha256']==verified[key]['measurement_sha256']
 goals=item['reviewed_goals']+([3] if r['new_ratio_with_published_tileops']<=2 else [])
 assert goals;r['timing_protocol']='v3-task-coverage-full32-20261103';r['attempt_version']=v;r['accepted_goals']=sorted(set(goals));r['review_note']=item['note'];r['baseline_improvement']=r['old_baseline_us']/r['new_baseline_us'];r['evidence_archive']=f'remote_evidence/evidence/{i:02d}-{v}.tar.gz';rows.append(r)
assert len(rows)==66 and len({r['ordinal'] for r in rows})==66
summary={'cases':66,'ops':len({r['op'] for r in rows}),'accepted':'each case meets at least one user criterion; original correctness + 15 v3 windows + valid guard + local raw replay required','within_2x':sum(r['new_ratio_with_published_tileops']<=2 for r in rows),'still_over_10x':sum(r['new_ratio_with_published_tileops']>10 for r in rows),'remaining_aicpu_cases':[r['ordinal'] for r in rows if r['aicpu_us']>0],'ratio_denominator':'frozen published TileOPs latency; TileOPs not remeasured in this round','median_baseline_improvement':statistics.median(r['baseline_improvement'] for r in rows),'goals_counts':{str(g):sum(g in r['accepted_goals'] for r in rows) for g in range(1,5)},'selected':rows}
(P/'FINAL_RESULTS.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
fields=['ordinal','op','case','attempt_version','timing_protocol','old_baseline_us','new_baseline_us','old_ratio','new_ratio_with_published_tileops','baseline_improvement','accepted_goals','aicpu_us','tasks_per_window','change','review_note','evidence_archive']
with (P/'RESULTS.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in rows)
print(json.dumps({k:v for k,v in summary.items() if k!='selected'},ensure_ascii=False,indent=2))

routes=[]
for r in rows:
 tasks='tasks' if r['version']=='v1' else 'tasks_'+r['version']
 task=json.loads((P/tasks/f"{r['ordinal']:02d}.json").read_text())
 routes.append({'key':[r['op'],r['case']],'module':pathlib.Path(task.get('candidate_file','candidates_v1.py')).stem,'measurement_sha256':r['measurement_sha256'],'accepted_goals':r['accepted_goals'],'evidence_archive':r['evidence_archive']})
(P/'ROUTES.json').write_text(json.dumps(routes,ensure_ascii=False,indent=2)+'\n')
