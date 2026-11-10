import json,pathlib,hashlib,struct,collections,statistics
R=pathlib.Path('/root/dyq/workspace');O=pathlib.Path(__file__).parent;P=R/'docs/ascend950/dev/norm_quant_local32_20261102'
ops=['INT8QuantPerTensorFwdOp','INT8QuantPerChannelFwdOp','INT8QuantPerBlockFwdOp','FP8QuantPerBlockFwdOp','SmoothQuantFwdOp','INT4QuantPerGroupFwdOp'];out=[]
for op in ops:
 label='int4-production-perf-v1' if op=='INT4QuantPerGroupFwdOp' else 'quant-production-perf-v1-'+op
 p=P/label;rows=[json.loads(l) for l in (p/'cases.jsonl').open()];cov=json.load(open(p/'raw-profiles/coverage.json'));gpath=P/(label+'-guard.json');g=json.load(open(gpath));errors=[]
 if not(g.get('performance_valid') is True and g.get('exit_code')==0 and not g.get('conflict_detected') and not g.get('visibility_lost')):errors.append('guard')
 if len(cov)!=30*len(rows):errors.append('coverage_count')
 idx=0
 for row in rows:
  d={'key':['torch-eager',op,row['case']],'source':str((p/'cases.jsonl').relative_to(R)),'guard':str(gpath.relative_to(R)),'eligible':row['eager_correctness'].get('eligible'),'eager_check':row['eager_correctness'],'errors':errors.copy(),'windows':[]}
  if row['runtime_limit']!={'cube_core_num':32,'vector_core_num':64}:d['errors'].append('cores')
  for label2 in ('tileops','eager'):
   t=row[label2];samples=t['latency_samples_ms']
   if abs(statistics.mean(samples)-t['latency_ms'])>1e-9:d['errors'].append('mean')
   for sample in samples:
    c=cov[idx];idx+=1;root=pathlib.Path(c['path']);h=hashlib.sha256();groups=collections.defaultdict(list)
    for f in sorted(root.rglob('*')):
     if not f.is_file() or f.name.endswith(('.done','.complete')):continue
     b=f.read_bytes();h.update(str(f.relative_to(root)).encode());h.update(hashlib.sha256(b).digest())
     if f.name.startswith('ffts_profile'):
      for v in struct.iter_unpack('<16q',b):
       if v[1]:groups[((v[0]>>32)&0xffffffff,(v[2]>>16)&65535)].append((v[14],v[15]))
    if h.hexdigest()!=c['complete_sha256']:d['errors'].append('full_digest')
    ns=sum(max(b for a,b in iv)-min(a for a,b in iv) for iv in groups.values());seqs={k[0]&65535 for k in groups};named=set();added=0.0
    for f in root.rglob('*task_track.slice_*'):
     if f.name.endswith(('.done','.complete')):continue
     for v in struct.iter_unpack('<8q',f.read_bytes()):
      if v[5]:named.add((v[3]>>32)&65535)
    covered=set(seqs)
    for f in root.rglob('stars_soc.data.*.slice_*'):
     if f.name.endswith(('.done','.complete')):continue
     info=json.load(open(f.parent.parent/'info.json.0'));mhz=float(info['DeviceInfo'][0]['hwts_frequency']);pending={}
     for v in struct.iter_unpack('<4HQHH3L',f.read_bytes()):
      fn=v[0]&63;k=(v[2]+(v[3]<<16),v[0]>>10)
      if fn not in (0,1):continue
      if fn==0:pending[k]=v[4]
      else:
       start=pending.pop(k)
       if k[0] in named and k[0] not in seqs:
        if k[1] not in (4,5):d['errors'].append('unexpected_nonffts_task')
        added+=(v[4]-start)/mhz;covered.add(k[0])
     if pending:d['errors'].append('unclosed_stars')
    if named-covered:d['errors'].append('uncovered_named_tasks')
    if abs(added-c['decoded']['added_us'])>1e-6:d['errors'].append('aicpu_decode_mismatch')
    # Extra task decoder is separately reviewed; record any dependence rather than claim independent FFTS proof.
    if abs(ns/1e6+added/1000-sample)>1e-9:d['errors'].append('raw_sample')
    d['windows'].append({'role':label2,'raw_ffts_ms':ns/1e6,'added_aicpu_us':added,'value_ms':sample})
  out.append(d)
(O/'QUANT_EXISTING_AUDIT.json').write_text(json.dumps(out,indent=2));print({'pairs':len(out),'eligible':sum(r['eligible'] is True for r in out),'errors':sum(bool(r['errors']) for r in out),'aicpu_windows':sum(w['added_aicpu_us']!=0 for r in out for w in r['windows'])})
