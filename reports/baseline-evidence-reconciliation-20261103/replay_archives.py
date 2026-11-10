"""CPU-only replay of every accepted archive; never mutates original receipts."""
import hashlib,importlib.util,json,pathlib,statistics,tarfile,tempfile,time
R=pathlib.Path('/root/dyq/workspace');O=pathlib.Path(__file__).parent;P=R/'reports/baseline-v3-full-20261102/server1'
spec=importlib.util.spec_from_file_location('parser',P/'v3/kernel_task_profile.py');parser=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser)
rows=[]
for tid,r in json.loads((P/'PROGRESS.json').read_text()).items():
 if r['status']!='accepted':continue
 a=pathlib.Path(r['archive']['local']);d={'key':r['key'],'task_id':tid,'archive':str(a),'errors':[],'windows':[]}
 try:
  assert hashlib.sha256(a.read_bytes()).hexdigest()==r['archive']['sha256'],'archive SHA'
  with tempfile.TemporaryDirectory(prefix='baseline-cpu-replay-') as tmp:
   dest=pathlib.Path(tmp)
   with tarfile.open(a) as t:
    for m in t.getmembers():
     rel=pathlib.PurePosixPath(m.name)
     assert not rel.is_absolute() and '..' not in rel.parts and (m.isfile() or m.isdir())
    t.extractall(dest,filter='data')
   for name,meta in json.loads((dest/'FILES.json').read_text())['files'].items():
    f=dest/name;assert f.stat().st_size==meta['size'] and hashlib.sha256(f.read_bytes()).hexdigest()==meta['sha256'],name
   m=json.loads((dest/'measurement.json').read_text());g=json.loads((dest/'guard.json').read_text());c=json.loads((dest/'correctness.json').read_text())
   assert g['performance_valid'] and g['exit_code']==g['workload_exit_code']==0
   assert c['status']=='verified' and c['inputs_preserved'] and not c.get('provider_source_changed') and not c.get('oracle_source_changed')
   assert hashlib.sha256((dest/'correctness.json').read_bytes()).hexdigest()==m['correctness_sha256']
   assert [m['identity']['environment'][k] for k in ('aicore_count','vectorcore_count')]==[32,64]
   assert m['key']==r['key'] and m['task_id']==tid and g['owner']=='v3-full/server1/'+tid
   assert len(m['timing']['retained_profiles'])==15
   for row in m['timing']['retained_profiles']:
    directory=dest/'profiles'/row['profile_path'].split('/profiles/',1)[1]
    parsed=parser.parse_profile(directory,1)
    d['windows'].append({'latency_ms':parsed['latency_ms'],'aicpu_count':parsed['profile_aicpu_task_count'],'trace_sha256':parser.trace_digest(directory),'tasks':parsed['profile_kernel_tasks']})
   d['old_ms']=m['timing']['latency_ms'];d['corrected_ms']=statistics.mean(w['latency_ms'] for w in d['windows']);d['changed']=abs(d['old_ms']-d['corrected_ms'])>1e-9
 except Exception as e:d['errors'].append(str(e))
 rows.append(d)
 if len(rows)%50==0:print(len(rows),'errors',sum(bool(x['errors']) for x in rows),flush=True)
(O/'REPLAY.json').write_text(json.dumps({'parser_revision':parser.PARSER_REVISION,'parser_sha256':hashlib.sha256((P/'v3/kernel_task_profile.py').read_bytes()).hexdigest(),'rows':rows},indent=2))
print('done',len(rows),'errors',sum(bool(x['errors']) for x in rows),'changed',sum(x.get('changed',False) for x in rows),flush=True)
