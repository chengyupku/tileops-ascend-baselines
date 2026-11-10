"""Select complete audited implementations, never a single fastest timing window."""
import pathlib,json,tarfile,collections
P=pathlib.Path(__file__).resolve().parent;D=P/'remote_evidence/evidence'
index=json.loads((D/'INDEX.json').read_text());audited={x['measurement_sha256'] for x in json.loads((P/'ARCHIVE_AUDIT.json').read_text())};replay={x['measurement_sha256']:x for x in json.loads((P/'remote_evidence/RAW_REPLAY.json').read_text())}
variants=[]
for out,entry in index.items():
 if 'control-' in out or entry['measurement_sha256'] not in audited:continue
 with tarfile.open(D/pathlib.Path(entry['archive']).name) as tar:
  m=json.load(tar.extractfile(out+'/measurement.json'));route=json.load(tar.extractfile(out+'/candidate_route.json'))
 raw=replay[entry['measurement_sha256']]
 variants.append({'key':m['key'],'out':out,'candidate_file':m['source_sha256']['candidate_file'],'measurement_sha256':entry['measurement_sha256'],'new_us':m['timing']['latency_ms']*1000,'aicpu_tasks_all15':sum(w['aicpu_tasks'] for w in raw['windows']),'route':route,'local_full_raw_verified':True})
groups=collections.defaultdict(list)
for r in variants:groups[tuple(r['key'])].append(r)
selected=[min(rows,key=lambda r:r['new_us']) for key,rows in sorted(groups.items())]
summary={'unique_cases':len(selected),'valid_variants':len(variants),'local_archives_including_control':len(audited),'local_raw_windows_including_control':15*len(audited),'selected_aicpu_free':sum(x['aicpu_tasks_all15']==0 for x in selected),'selection':'lowest complete mean15 among independently correct and guard-valid implementations; never select individual windows; comparisons not necessarily interleaved A/B','ops':dict(collections.Counter(x['key'][1] for x in selected)),'sampling_dtype_correction':'SAMPLING_DTYPE_CORRECTION.json applies to all sampling variants; original NPU .double() actually FP32'}
for name,data in [('CURRENT_VARIANTS.json',variants),('CURRENT_RESULTS.json',selected),('CURRENT_SUMMARY.json',summary)]:
 temp=P/(name+'.tmp');temp.write_text(json.dumps(data,indent=2));temp.replace(P/name)
print(json.dumps(summary,indent=2))
for key,rows in sorted(groups.items()):
 if len(rows)>1:print('comparison',key,[(x['out'],round(x['new_us'],3),x['aicpu_tasks_all15']) for x in rows])
