import json,pathlib,struct,statistics,datetime
root=pathlib.Path('/opt/tileops-ci/server2-20261030/overnight-20261101/ci-full-v3/measurements')
rows=[];errors=[]
for p in sorted(root.rglob('measurements.jsonl')):
 for line in p.read_text().splitlines():
  try:r=json.loads(line)
  except ValueError:continue
  for typ,x in [('candidate',r),('eager',r.get('eager',{}))]:
   if not isinstance(x,dict) or not x.get('latency_ms'):continue
   item={'op':r.get('op'),'case_id':r.get('case_id'),'implementation':typ,'source':str(p),'has_H':'end_to_end_mean_ms' in x,'v3_ms':x['latency_ms']}
   try:
    values=[];mixed=False
    for prof in x.get('retained_profiles',[]):
     tasks=prof['profile_kernel_tasks']; keys={(t['task_id'],t['context_id']) for t in tasks}
     mixed |= any(len(t['core_roles'])>1 for t in tasks)
     total=0;files=list(pathlib.Path(prof['profile_path']).rglob('ffts_profile*'))
     if not files:raise ValueError('raw trace absent')
     for f in files:
      if f.name.endswith(('.done','.complete')):continue
      raw=f.read_bytes()
      for off in range(0,len(raw),128):
       v=struct.unpack_from('<16q',raw,off)
       if v[1] and ((v[0]>>32)&0xFFFFFFFF,(v[2]>>16)&65535) in keys:total+=v[15]-v[14]
     values.append(total/1e6)
    if len(values)!=15:raise ValueError('not 15 profiles')
    item.update(v2_reparsed_ms=statistics.mean(values),mixed_same_task=mixed,ratio=statistics.mean(values)/x['latency_ms'])
   except Exception as e:errors.append({'source':str(p),'error':str(e)})
   rows.append(item)
def dist(xs):
 xs=sorted(xs)
 if not xs:return {}
 def q(p):
  a=(len(xs)-1)*p;i=int(a);return xs[i]+(xs[min(i+1,len(xs)-1)]-xs[i])*(a-i)
 return dict(n=len(xs),min=xs[0],median=q(.5),p90=q(.9),max=xs[-1])
print(json.dumps({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rows':rows,'errors':errors,'summary':{'measured_rows':len(rows),'with_H':sum(r['has_H'] for r in rows),'single_role_v2_over_v3':dist([r['ratio'] for r in rows if 'ratio' in r and not r['mixed_same_task']]),'mixed_v2_over_v3':dist([r['ratio'] for r in rows if 'ratio' in r and r['mixed_same_task']])}},indent=2))
