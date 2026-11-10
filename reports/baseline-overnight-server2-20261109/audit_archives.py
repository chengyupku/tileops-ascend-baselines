"""Local, CPU-only validation of downloaded archives and raw timing records."""
import pathlib,json,hashlib,tarfile,tempfile,importlib.util,math
P=pathlib.Path(__file__).resolve().parent;D=P/'remote_evidence/evidence'
spec=importlib.util.spec_from_file_location('parser',P.parent/'torch-v010-20261106/tools/kernel_task_profile_full32.py');parser=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser)
index=json.loads((D/'INDEX.json').read_text());report=P/'ARCHIVE_AUDIT.json';rows=json.loads(report.read_text()) if report.exists() else [];known={r['archive_sha256'] for r in rows}
for key,e in index.items():
 archive=D/pathlib.Path(e['archive']).name
 with archive.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 assert h==e['archive_sha256']
 if h in known:continue
 with tarfile.open(archive) as tar, tempfile.TemporaryDirectory(prefix='torch66-replay-') as tmp:
  def raw(name):return tar.extractfile(name).read()
  mbytes=raw(key+'/measurement.json');assert hashlib.sha256(mbytes).hexdigest()==e['measurement_sha256'];m=json.loads(mbytes)
  cb=raw(key+'/correctness.json');assert hashlib.sha256(cb).hexdigest()==m['correctness_sha256'];correct=json.loads(cb);assert correct['status']=='verified' and correct['inputs_preserved']
  proof=m['source_sha256'];assert hashlib.sha256(raw('implementation/'+proof['candidate_file'])).hexdigest()==proof['candidate_sha256']
  wh=[hashlib.sha256(raw(n)).hexdigest() for n in tar.getnames() if n.startswith('implementation/worker')];assert proof['worker_sha256'] in wh
  for version in ['v1','v2','v4']:
   dk='candidate_dependency_'+version+'_sha256'
   if dk in proof:assert hashlib.sha256(raw('implementation/candidates_'+version+'.py')).hexdigest()==proof[dk]
  guards=[raw(n) for n in tar.getnames() if n.startswith(key+'/guard') and n.endswith('.json')]
  guard=next(json.loads(b) for b in guards if hashlib.sha256(b).hexdigest()==e['guard_sha256']);assert guard['performance_valid'] and guard['phase']=='finished' and guard['workload_exit_code']==0
  for member in tar.getmembers():
   if member.isfile() and '/profiles/' in member.name and any(s in member.name for s in ['ffts_profile','stars_soc.data.','hash_dic.slice_','task_track.slice_','info.json']):tar.extract(member,tmp,filter='data')
  timing=m['timing'];windows=[]
  for pr,latency in zip(timing['retained_profiles'],timing['latency_samples_ms'],strict=True):
   root=pathlib.Path(tmp)/key/'profiles'/pr['profile_path'].split('/profiles/',1)[1]
   assert parser.trace_digest(root)==pr['profile_trace_sha256'];r=parser.parse_profile(root,1)
   assert r['profile_kernel_tasks']==pr['profile_kernel_tasks'];assert math.isclose(r['latency_ms'],latency,abs_tol=1e-7)
   windows.append(pr['profile_trace_sha256'])
  assert len(windows)==15 and m['repeats']==15 and m['warmup']==3
  assert math.isclose(sum(timing['latency_samples_ms'])/15,timing['latency_ms'],abs_tol=1e-7)
 rows.append({'key':key,'archive_sha256':h,'measurement_sha256':e['measurement_sha256'],'status':'source/correctness/guard/all 15 raw windows verified locally','trace_sha256':windows});report.write_text(json.dumps(rows,indent=2));print(key,flush=True)
print('locally audited archives:',len(rows),flush=True)
