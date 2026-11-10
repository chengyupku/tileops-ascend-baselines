import json,pathlib,hashlib,tarfile,tempfile,statistics,sys
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P))
from tools.kernel_task_profile_full32 import parse_profile,trace_digest
rows=[]
for host in ['server1','server2']:
 archive=P/host/'rope-a1.tar.gz';expected=(P/host/'rope-archive-sha.txt').read_text().split()[0]
 assert hashlib.sha256(archive.read_bytes()).hexdigest()==expected
 with tempfile.TemporaryDirectory() as tmp:
  root=pathlib.Path(tmp)
  with tarfile.open(archive) as t:t.extractall(root,filter='data')
  m=json.load(open(root/'measurement.json'));g=json.load(open(root/'guard.json'));c=json.load(open(root/'correctness.json'))
  assert g['performance_valid'] and g['exit_code']==0 and c['status']=='verified'
  values=[];raw_names=set()
  for r,v in zip(m['timing']['retained_profiles'],m['timing']['latency_samples_ms']):
   rel=pathlib.Path(r['profile_path']).relative_to('/opt/torch-v010-20261106/preflight/rope-a1');d=root/rel
   assert trace_digest(d)==r['profile_trace_sha256'];z=parse_profile(d,1);assert abs(z['latency_ms']-v)<1e-9;values.append(v)
   for f in d.rglob('*hash_dic.slice_*'):
    if not f.name.endswith(('.done','.complete')):raw_names.update(f.read_text().splitlines())
  assert len(values)==15
  rows.append({'host':host,'guard_valid':True,'correctness':c,'raw_windows':15,'archive_sha256':expected,'mean_ms':statistics.mean(values),'raw_name_dictionary':sorted(raw_names),'boundary':json.load(open(root/'timing_boundary.json'))})
(P/'PREFLIGHT_ROPE_REPLAY.json').write_text(json.dumps(rows,indent=2));print([(r['host'],r['mean_ms'],r['raw_windows']) for r in rows])
