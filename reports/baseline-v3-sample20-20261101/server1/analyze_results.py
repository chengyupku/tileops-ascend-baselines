import json,pathlib,statistics,struct,datetime
P=pathlib.Path(__file__).resolve().parent
sample=json.loads((P/'SAMPLE.json').read_text());out=[]
for old in sample['rows']:
 paths=sorted((P/'runs'/old['task_id']).glob('*/measurement.json'))
 for p in reversed(paths):
  m=json.loads(p.read_text());g=p.parent/'guard.json'
  if old['task_id'] in ('068418bab638b1a98166','dd73beb2e0ad1008d0d6') and p.parent.name=='a1':continue
  if m.get('status')!='measured_pending_guard' or not g.exists() or not json.loads(g.read_text()).get('performance_valid'):continue
  guard=json.loads(g.read_text());guard_seconds=(datetime.datetime.fromisoformat(guard['updated_at'])-datetime.datetime.fromisoformat(guard['acquired_at'])).total_seconds()
  t=m['timing'];profiles=t.get('retained_profiles',[]);d2=[];d3=[]
  for prof in profiles:
   rel=pathlib.Path(prof['profile_path']).relative_to(pathlib.Path(str(p.parent).replace(str(P),'/opt/tileops-baseline-cache/workspace/reports/baseline-v3-sample20-20261101/server1')))
   records=[]
   for f in (p.parent/rel).rglob('ffts_profile*'):
    if f.name.endswith(('.done','.complete')):continue
    raw=f.read_bytes()
    for i in range(0,len(raw),128):
     v=struct.unpack_from('<16q',raw,i)
     if v[1]:records.append(v)
   keys={(x['task_id'],x['context_id']) for x in prof['profile_kernel_tasks']}
   d2.append(sum(v[15]-v[14] for v in records if ((v[0]>>32)&0xffffffff,(v[2]>>16)&65535) in keys)/1e6)
   d3.append(sum(x['duration_us'] for x in prof['profile_kernel_tasks'])/1000)
  assert len(d2)==15
  v3=t['latency_ms'];assert abs(statistics.mean(d3)-v3)<1e-6
  row={'key':old['key'],'historical_host':old['host'],'stratum':old['sampling_stratum'],'population_weight':old['stratum_population'],'old_v2_ms':old['timing']['latency_ms'],'old_H_ms':old['end_to_end_mean_ms'],'new_v3_ms':v3,'new_v2_ms':statistics.mean(d2),'new_H_ms':m['end_to_end_mean_ms'],'measurement_wall_seconds':t['measurement_wall_seconds'],'full_wall_seconds':m['experiment_wall_seconds'],'guard_elapsed_seconds':guard_seconds,'receipt':str(p)}
  for name in ['old_v2','old_H','new_v2','new_H']:row[name+'_absolute_relative_error']=abs(row[name+'_ms']/v3-1)
  out.append(row);break
report={'completed':len(out),'sample_size':20,'extrapolation_complete':len(out)==20,'rows':out}
if out:
 report['mean_absolute_relative_error']={name:statistics.mean(r[name+'_absolute_relative_error'] for r in out) for name in ['old_v2','old_H','new_v2','new_H']}
 report['weighted_mean_absolute_relative_error']={name:sum(r[name+'_absolute_relative_error']*r['population_weight'] for r in out)/sum(r['population_weight'] for r in out) for name in ['old_v2','old_H','new_v2','new_H']}
 report['extrapolation_hours' if len(out)==20 else 'partial_strata_hours_NOT_full_forecast']={name:sum(r[name]*r['population_weight'] for r in out)/3600 for name in ['measurement_wall_seconds','full_wall_seconds','guard_elapsed_seconds']}
(P/'RESULTS.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
for r in out:print(r['key'],round(r['new_v3_ms'],5),'v2err',round(100*r['old_v2_absolute_relative_error'],1),'Herr',round(100*r['old_H_absolute_relative_error'],1))
