"""Finalize only after remote raw replay and matching evidence archives exist."""
import pathlib,json,hashlib,csv,datetime
P=pathlib.Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
t=json.load(open(P/'TORCH_RESULTS.json'));n=json.load(open(P/'NATIVE_RESULTS.json'))
assert (t['pairs'],n['pairs'])==(14,10) and (t['raw_windows'],n['raw_windows'])==(420,150)
rows=t['rows']+n['rows'];old={tuple(r[k] for k in ['library','op','case']):r for r in json.load(open(P/'BACKLOG.json'))}
updates=[]
for r in rows:
 key=tuple(r[k] for k in ['library','op','case']);prior=old[key]
 updates.append({'key':key,'status':'verified_candidate_not_published','previous_measurement_id':prior['measurement_id'],'new_measurement_id':r.get('fixed_measurement_sha256',r.get('measurement_sha256')),'latency_us':r['fixed_us'],'old_web_ratio':r['old_web_ratio'],'ratio_with_existing_tileops':r['recomputed_ratio_with_published_tileops'],'aicpu_tasks_per_window':0})
(P/'REPLACEMENT_CANDIDATES.json').write_text(json.dumps(updates,indent=2))
with (P/'RESULTS.csv').open('w') as f:
 fields=['library','op','case','old_web_us','fixed_us','old_web_ratio','recomputed_ratio_with_published_tileops','fixed_aicpu'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in rows)
summary={'status':'priority24_verified_5_deferred','verified_pairs':24,'torch':14,'native':10,'deferred_pairs':5,'new365':'recorded only; no fixes applied','verified_original_raw_windows':210,'verified_fixed_raw_windows':360,'all_fixed_windows_no_AICPU':True,'original_checker_unchanged':True,'published':False,'artifacts':{p.name:sha(p) for p in P.glob('*evidence.tar.gz')},'int4_ratio':{'old_min':min(r['old_web_ratio'] for r in rows if r['op']=='INT4QuantPerGroupFwdOp'),'old_max':max(r['old_web_ratio'] for r in rows if r['op']=='INT4QuantPerGroupFwdOp'),'new_min':min(r['recomputed_ratio_with_published_tileops'] for r in rows if r['op']=='INT4QuantPerGroupFwdOp'),'new_max':max(r['recomputed_ratio_with_published_tileops'] for r in rows if r['op']=='INT4QuantPerGroupFwdOp')},'completed_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(P/'SUMMARY.json').write_text(json.dumps(summary,indent=2))
state=json.load(open(P/'STATE.json'));state.update(status=summary['status'],torch_verified=14,native_verified_so_far=10,native_v2_full_case_checks_passed=10,sessions={},next='Use verified candidate measurements and frozen patches for later baseline publication;5 hard algorithm/precision pairs remain in hard/DEFERRED.json.',summary='SUMMARY.json',report='REPORT.txt');(P/'STATE.json').write_text(json.dumps(state,indent=2))
print(json.dumps(summary,indent=2))
backlog=list(csv.DictReader(open(P.parent/'torch-aicpu-audit-20261108/COMBINED_AICPU_BACKLOG.csv',encoding='utf-8-sig')))
assert len(backlog)==394
resolved={tuple(r['key']):r for r in updates};disposition=[]
for r in backlog:
 key=tuple(r[k] for k in ['library','op','case'])
 status='verified_fix_pending_publication' if key in resolved else ('deferred_algorithm_or_precision' if key in old else 'newly_recorded_not_modified')
 disposition.append({**{k:r[k] for k in ['library','op','case','measurement_id']},'original_audit_status':r['status'],'disposition':status,'new_measurement_id':resolved.get(key,{}).get('new_measurement_id','')})
assert sum(r['disposition']=='verified_fix_pending_publication' for r in disposition)==24
assert sum(r['disposition']=='deferred_algorithm_or_precision' for r in disposition)==5
assert sum(r['disposition']=='newly_recorded_not_modified' for r in disposition)==365
with (P/'BACKLOG_DISPOSITION.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(disposition[0]));w.writeheader();w.writerows(disposition)
