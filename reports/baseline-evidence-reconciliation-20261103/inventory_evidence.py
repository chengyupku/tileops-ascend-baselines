"""Read every located newer eager record and its retained hardware evidence, CPU only."""
import json,pathlib,hashlib,collections
R=pathlib.Path('/root/dyq/workspace');A=R/'reports/baseline-independent-review-20261103';O=pathlib.Path(__file__).parent
source_cache={};coverage_cache={};out=[]
for e in json.loads((A/'NEWER_EAGER_EVIDENCE.json').read_text()):
 source=R/e['source']
 if source not in source_cache:source_cache[source]=[json.loads(l) for l in source.read_text().splitlines() if l.strip()]
 row=source_cache[source][e['line']-1]; t=row.get('eager',{}); ev=t.get('profile_coverage_evidence')
 coverage=pathlib.Path(ev) if ev else source.parent/'raw-profiles/coverage.json'
 if not coverage.is_absolute():coverage=R/coverage
 if coverage not in coverage_cache:
  facts={'coverage_path':str(coverage),'exists':coverage.exists(),'hardware_counts':{},'missing_profile_roots':0,'info_files':0}
  if coverage.exists():
   facts['sha256']=hashlib.sha256(coverage.read_bytes()).hexdigest()
   try:
    cov=json.loads(coverage.read_text());hw=collections.Counter()
    for w in cov:
     root=pathlib.Path(w['path']);root=root if root.is_absolute() else R/root
     if not root.exists():facts['missing_profile_roots']+=1;continue
     for f in root.rglob('info.json.0'):
      d=json.loads(f.read_text());v=d['DeviceInfo'][0];hw[(v.get('ai_core_num'),v.get('aiv_num'),d.get('hostname'))]+=1;facts['info_files']+=1
    facts['hardware_counts']={'/'.join(map(str,k)):v for k,v in hw.items()}
   except Exception as ex:facts['error']=str(ex)
  coverage_cache[coverage]=facts
 facts=coverage_cache[coverage]
 reasons=[]
 if t.get('measurement_protocol')!='ascend950-isolated-ffts-v3':reasons.append('older_timing_protocol')
 if row.get('eager_correctness',{}).get('eligible') is not True:reasons.append('eager_correctness_not_explicitly_eligible')
 if not facts['exists'] or not facts['info_files']:reasons.append('retained_hardware_evidence_missing')
 elif any(not k.startswith('32/64/') for k in facts['hardware_counts']):reasons.append('physical_hardware_not_32AIC_64AIV')
 if facts.get('missing_profile_roots'):reasons.append('raw_profile_missing')
 out.append({'key':e['key'],'source':e['source'],'line':e['line'],'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'protocol':t.get('measurement_protocol'),'runtime_limit':row.get('runtime_limit'),'correctness':row.get('eager_correctness'),'coverage':str(coverage),'admission_blockers':reasons,'admitted':False})
(O/'NEWER_RECORDS.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));(O/'RAW_HARDWARE.json').write_text(json.dumps(list(coverage_cache.values()),indent=2))
print('records',len(out),'coverage_files',len(coverage_cache),'no_initial_blockers',sum(not x['admission_blockers'] for x in out),'reasons',dict(collections.Counter(z for x in out for z in x['admission_blockers'])),flush=True)
