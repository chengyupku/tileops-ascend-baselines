import json,hashlib,statistics,datetime,copy
from pathlib import Path
P=Path('reports/aicpu-baseline-fix-server2-20261108');O=Path('reports/baseline24-publication-20261108');S=Path('/tmp/baseline24-publish-20261108')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
summary=read(P/'SUMMARY.json');assert summary['verified_pairs']==24
for name,h in summary['artifacts'].items():assert sha(P/name)==h
cs=read(P/'REPLACEMENT_CANDIDATES.json');assert len(cs)==24
files=list((P/'torch-evidence/runs').glob('*/fixed/measurement.json'))+list((P/'native-evidence').rglob('measurement.json'));byhash={sha(f):f for f in files}
snap=S/'data/baseline-cache/snapshot.json';O.joinpath('snapshot.before.json').write_bytes(snap.read_bytes());d=read(snap);rows={tuple(r[k] for k in ['library','op','case']):r for r in d['rows']};replaced=[];audit=[]
for c in cs:
 k=tuple(c['key']);r=rows[k];assert r['measurement_id']==c['previous_measurement_id'],k
 f=byhash[c['new_measurement_id']];m=read(f);t=m['timing'];assert len(t['retained_profiles'])==len(t['latency_samples_ms'])==15
 assert all(q['profile_aicpu_task_count']==0 and q['profile_named_task_coverage_complete'] for q in t['retained_profiles'])
 assert abs(statistics.mean(t['latency_samples_ms'])*1000-c['latency_us'])<1e-7
 corr=f.parent/'correctness.json' if k[0]!='torch-eager' else f.parent.parent/'correctness.json';assert sha(corr)==m['correctness_sha256']
 proof=read(corr);case=proof.get('case_snapshot',{})
 if case:assert case['parameters']==r['workload']['params'],k
 if k[0]!='torch-eager':
  g=read(f.parent/'guard.json');assert g['performance_valid'] and g['workload_exit_code']==0 and g['phase']=='finished'
  assert m['status'] in ['measured','measured_pending_guard'] and proof['status']=='verified'
  expected=next(x for x in read(P/'NATIVE_RESULTS.json')['rows'] if (x['library'],x['op'],x['case'])==k)
  assert sha(f.parent/'guard.json')==expected['guard_sha256']
 replaced.append(copy.deepcopy(r));r.update(latency_us=c['latency_us'],measurement_id=c['new_measurement_id'],protocol=t['measurement_protocol'],parser_revision=t['parser_revision'],samples=15)
 r['ratio']=r['latency_us']/r['tileops_latency_us'] if r.get('tileops_latency_us') else None
 audit.append({'key':k,'measurement':str(f),'sha256':sha(f),'correctness_sha256':sha(corr),'old_id':c['previous_measurement_id'],'new_latency_us':r['latency_us']})
now=datetime.datetime.now(datetime.timezone.utc).isoformat();d['frozen_at']=now;d['baseline_release']='verified-aicpu24-20261108';d.setdefault('publication_notes',[]).append('2026-11-08: replace 14 torch-eager and 10 tilelang-ascend baselines with verified AICPU-free device measurements; workload contracts and existing TileOPs measurements retained.')
snap.write_text(json.dumps(d,ensure_ascii=False,separators=(',',':'))+'\n')
release=read(S/'data/baseline-cache/latest-release.json');release.update(id=d['baseline_release'],snapshot_sha256=sha(snap));release['measurements']=[{k:r[k] for k in ['library','op','case','measurement_id','protocol']} for r in d['rows']];(S/'data/baseline-cache/latest-release.json').write_text(json.dumps(release,ensure_ascii=False,separators=(',',':'))+'\n')
history={'release':d['baseline_release'],'replaced_rows':replaced,'replacements':cs,'evidence_archive_sha256':summary['artifacts'],'note':'Superseded measurements retained for audit; not current comparison candidates.'};(S/'data/baseline-cache/aicpu24-correction-20261108.json').write_text(json.dumps(history,ensure_ascii=False,indent=2)+'\n');O.joinpath('AUDIT.json').write_text(json.dumps({'count':24,'rows':audit,'raw_replay_boundary':'Original collector replayed570windows; publication verifies immutable archives, measurement and correctness hashes,15samples and profile metadata. No new device measurements.'},indent=2))
print('24 replacements verified; SHA',sha(snap))
