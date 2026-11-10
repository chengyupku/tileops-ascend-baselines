import json,pathlib,hashlib,tarfile,tempfile,statistics,sys
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P))
from tools.kernel_task_profile_full32 import parse_profile,trace_digest
hosts=sys.argv[1:] or ['server1','server2']
assert set(hosts)<= {'server1','server2'}
rows=[]
for host in hosts:
 archive=P/host/'final-representatives.tar.gz';expected=(P/host/'final-representatives-sha.txt').read_text().split()[0]
 assert hashlib.sha256(archive.read_bytes()).hexdigest()==expected
 with tempfile.TemporaryDirectory() as tmp:
  root=pathlib.Path(tmp)
  with tarfile.open(archive) as t:t.extractall(root,filter='data')
  for case in sorted(root.iterdir()):
   m=json.load(open(case/'measurement.json'));g=json.load(open(case/'guard.json'));c=json.load(open(case/'correctness.json'))
   assert g['performance_valid'] and g['exit_code']==0 and c['status']=='verified'
   assert g['phase']=='finished' and g['workload_exit_code']==0
   assert (m['hardware']['cube'],m['hardware']['vector'])==(32,64)
   assert m['source_sha256'] in {hashlib.sha256(f.read_bytes()).hexdigest() for f in P.glob('SOURCES.v*.json')}
   log=(case/'device.log').read_text(errors='replace')
   assert 'fall back to run on the CPU' not in log and 'npu_cpu_fallback' not in log
   profiles=m['timing']['retained_profiles'];samples=m['timing']['latency_samples_ms']
   assert len(profiles)==len(samples)==15
   names=set()
   for r,v in zip(profiles,samples):
    rel=pathlib.Path(r['profile_path']).relative_to('/opt/torch-v010-20261106/preflight/'+case.name);d=case/rel
    assert trace_digest(d)==r['profile_trace_sha256'];z=parse_profile(d,1);assert abs(z['latency_ms']-v)<1e-9
    for f in d.rglob('*hash_dic.slice_*'):
     if not f.name.endswith(('.done','.complete')):names.update(f.read_text().splitlines())
   rows.append({'host':host,'attempt':case.name,'task':json.load(open(case/'task.json')),'correctness':c,'guard_valid':True,'raw_windows':15,'archive_sha256':expected,'mean_ms':statistics.mean(samples),'raw_name_dictionary':sorted(names),'boundary':json.load(open(case/'timing_boundary.json'))})
assert len(rows)==sum(3 if h=='server1' else 4 for h in hosts)
(P/('PREFLIGHT_FINAL_REPLAY'+('' if len(hosts)==2 else '-'+hosts[0])+'.json')).write_text(json.dumps(rows,indent=2));print([(r['host'],r['attempt'],r['mean_ms']) for r in rows])
