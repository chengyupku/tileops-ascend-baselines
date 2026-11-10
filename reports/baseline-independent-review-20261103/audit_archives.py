"""Independent CPU-only byte audit; writes only beside this script."""
import json,pathlib,hashlib,tarfile,struct,collections,statistics
ROOT=pathlib.Path('/root/dyq/workspace');OUT=pathlib.Path(__file__).parent;P=ROOT/'reports/baseline-v3-full-20261102/server1'
progress=json.load(open(P/'PROGRESS.json'));results=[]
def sha(b):return hashlib.sha256(b).hexdigest()
def union_ns(ivals):
 total=0;end=0
 for a,b in sorted(ivals):
  total+=max(0,b-max(a,end));end=max(end,b)
 return total
for tid,r in progress.items():
 if r['status']!='accepted':continue
 ap=pathlib.Path(r['archive']['local']);d={'key':r['key'],'task_id':tid,'archive':str(ap.relative_to(ROOT)),'errors':[],'windows':[]}
 try:
  if sha(ap.read_bytes())!=r['archive']['sha256']:d['errors'].append('archive_sha')
  with tarfile.open(ap) as t:
   files={m.name.removeprefix('./'):t.extractfile(m).read() for m in t if m.isfile()}
  manifest=json.loads(files['FILES.json'])['files']
  for name,meta in manifest.items():
   if name not in files or len(files[name])!=meta['size'] or sha(files[name])!=meta['sha256']:d['errors'].append('file_hash:'+name)
  m=json.loads(files['measurement.json']);c=json.loads(files['correctness.json']);g=json.loads(files['guard.json']);task=json.loads(files['task.json'])
  d['guard_scope']=g.get('coordination_scope');d['core_counts']=[m['identity']['environment'].get('aicore_count'),m['identity']['environment'].get('vectorcore_count')]
  d['source_version']=m.get('experiment_v3_sources');d['metric']=m.get('comparison_metric');d['correctness_status']=c.get('status');d['input_preserved']=c.get('inputs_preserved')
  for label,ok in {'key':m['key']==r['key']==task['key'],'task_id':m['task_id']==tid,'guard':g.get('performance_valid') is True and g.get('exit_code')==g.get('workload_exit_code')==0,'guard_owner':g.get('owner')=='v3-full/server1/'+tid,'correctness':c.get('status')=='verified' and c.get('inputs_preserved') and not c.get('provider_source_changed') and not c.get('oracle_source_changed'),'correctness_hash':sha(files['correctness.json'])==m['correctness_sha256'],'cores':d['core_counts']==[32,64],'protocol':m['identity']['protocol']['id']=='ascend950-isolated-ffts-v3'}.items():
   if not ok:d['errors'].append(label)
  rows=m['timing']['retained_profiles'];samples=m['timing']['latency_samples_ms']
  if len(rows)!=15 or len(samples)!=15:d['errors'].append('sample_count')
  if len({x['profile_path'] for x in rows})!=15:d['errors'].append('duplicate_path')
  for i,row in enumerate(rows):
   prefix='profiles/'+row['profile_path'].split('/profiles/',1)[1]+'/'
   selected={n[len(prefix):]:b for n,b in files.items() if n.startswith(prefix)}
   h=hashlib.sha256()
   for n,b in sorted(selected.items()):
    if not n.endswith(('.done','.complete')) and any(x in pathlib.Path(n).name for x in ('ffts_profile','hash_dic.slice_','task_track.slice_')):h.update(n.encode());h.update(hashlib.sha256(b).digest())
   if h.hexdigest()!=row['profile_trace_sha256']:d['errors'].append('trace_digest:'+str(i))
   groups=collections.defaultdict(list);raw_ns=0;raw_count=0;devices=set()
   for n,b in selected.items():
    if pathlib.Path(n).name.startswith('ffts_profile') and not n.endswith(('.done','.complete')):
     if len(b)%128:d['errors'].append('truncated')
     devices.add(n.split('/data/')[0])
     for v in struct.iter_unpack('<16q',b):
      if not v[1]:continue
      if v[0]&63!=42 or not 0<v[14]<=v[15]:d['errors'].append('raw_header_time')
      role='AIV' if v[2]>=1<<31 else 'AIC';k=((v[0]>>32)&0xffffffff,(v[2]>>16)&65535)
      groups[k].append((role,v[14],v[15]));raw_ns+=v[15]-v[14];raw_count+=1
   hashes={};names={};stars=[];star_errors=[]
   for n,b in selected.items():
    if 'hash_dic.slice_' in n and not n.endswith(('.done','.complete')):
     for line in b.decode(errors='replace').splitlines():
      v=line.split(':',1)
      if len(v)==2:
       try:
        hv=int(v[0]);hashes[hv-(1<<64) if hv>=1<<63 else hv]=v[1]
       except ValueError:pass
   for n,b in selected.items():
    if 'task_track.slice_' in n and not n.endswith(('.done','.complete')):
     for v in struct.iter_unpack('<8q',b):
      if v[5]:names[(v[3]>>32)&65535]=hashes.get(v[5],'<unknown>')
    if pathlib.Path(n).name.startswith('stars_soc.data.') and not n.endswith(('.done','.complete')):
     info=json.loads(selected[str(pathlib.PurePosixPath(n).parent.parent/'info.json.0')]);mhz=float(info['DeviceInfo'][0]['hwts_frequency']);pending={}
     if str(info['platform_version'])!='15':star_errors.append('platform')
     for v in struct.iter_unpack('<4HQHH3L',b):
      fn=v[0]&63
      if fn not in (0,1):continue
      sk=(v[2]+(v[3]<<16),v[0]>>10)
      if fn==0:pending[sk]=v[4]
      elif sk in pending:
       start=pending.pop(sk);stars.append({'seq':sk[0],'type':sk[1],'start':start,'end':v[4],'duration_us':(v[4]-start)/mhz})
      else:star_errors.append('unmatched_end')
     if pending:star_errors.append('unmatched_start')
   seqs={k[0]&65535 for k in groups}
   missing={k:v for k,v in names.items() if k not in seqs}
   additions=[dict(t,name=names[t['seq']]) for t in stars if t['seq'] in missing]
   spans=[(min(v[1] for v in vs),max(v[2] for v in vs)) for vs in groups.values()]
   total=sum(b-a for a,b in spans);uni=union_ns(spans)
   if abs(total/1e6-samples[i])>1e-9:d['errors'].append('sample_value:'+str(i))
   stored=sum(x['duration_us'] for x in row['profile_kernel_tasks'])/1000
   if abs(stored-total/1e6)>1e-9:d['errors'].append('stored_tasks:'+str(i))
   repeated=sum(any(len({(a,b) for r,a,b in vs if r==role})>1 for role in ['AIC','AIV']) for vs in groups.values())
   if repeated:d['errors'].append('repeated_task_context')
   d['windows'].append({'index':i,'named_uncovered_by_ffts':missing,'missing_named_stars':additions,'stars_errors':star_errors,'raw_records':raw_count,'tasks':len(groups),'dual_role_tasks':sum(len({v[0] for v in vs})==2 for vs in groups.values()),'raw_sum_ns':raw_ns,'task_span_sum_ns':total,'task_interval_union_ns':uni,'overlap_ns':total-uni,'devices':len(devices),'trace_sha':h.hexdigest()})
  if abs(statistics.mean(samples)-m['timing']['latency_ms'])>1e-9:d['errors'].append('mean')
 except Exception as e:d['errors'].append(repr(e))
 results.append(d)
 if len(results)%50==0:print('audited',len(results),flush=True)
(OUT/'ARCHIVE_AUDIT.json').write_text(json.dumps(results,indent=2))
summary={'pairs':len(results),'errors':sum(bool(r['errors']) for r in results),'errors_counter':dict(collections.Counter(x for r in results for x in r['errors'])),'dual_role_pairs':sum(any(w['dual_role_tasks'] for w in r['windows']) for r in results),'cross_task_overlap_pairs':sum(any(w['overlap_ns'] for w in r['windows']) for r in results),'guard_scopes':dict(collections.Counter(r.get('guard_scope') for r in results))}
(OUT/'ARCHIVE_SUMMARY.json').write_text(json.dumps(summary,indent=2));print(summary)
