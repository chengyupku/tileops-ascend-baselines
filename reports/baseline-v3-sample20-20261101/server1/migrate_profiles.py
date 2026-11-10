import pathlib,json,hashlib,subprocess,time
P=pathlib.Path(__file__).resolve().parent;R='/opt/tileops-baseline-cache/workspace/reports/baseline-v3-sample20-20261101/server1';jobs=[]
for g in P.glob('runs/*/*/guard.json'):
 guard=json.loads(g.read_text());profile=g.parent/'profiles'
 if not guard.get('workload_completed') or not profile.exists():continue
 hashes={str(f.relative_to(profile)):hashlib.sha256(f.read_bytes()).hexdigest() for f in profile.rglob('*') if f.is_file()}
 if hashes:jobs.append({'relative':str(g.parent.relative_to(P)),'hashes':hashes})
code='''import pathlib,json,hashlib,shutil,time
R=pathlib.Path(ROOT_VALUE);jobs=JOBS_VALUE;results=[]
for j in jobs:
 base=R/j['relative'];p=base/'profiles'
 if not p.exists():continue
 guard=json.loads((base/'guard.json').read_text())
 assert guard.get('workload_completed')
 assert str(p.resolve()).startswith(str(R/'runs')+'/')
 actual={str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in p.rglob('*') if f.is_file()}
 assert actual==j['hashes'], 'local/remote profile mismatch'
 receipt={'at':time.time(),'files':len(actual),'bytes':sum(f.stat().st_size for f in p.rglob('*') if f.is_file()),'sha256':actual,'destination':'verified local experiment runs/'+j['relative']+'/profiles'}
 (base/'MIGRATED_PROFILES.json').write_text(json.dumps(receipt,indent=2))
 shutil.rmtree(p);results.append({'relative':j['relative'],**receipt})
print(json.dumps({'at':time.time(),'free_GiB':shutil.disk_usage(R).free/2**30,'migrations':results}))
'''.replace('ROOT_VALUE',repr(R)).replace('JOBS_VALUE',repr(jobs))
r=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ClearAllForwardings=yes','openlibing-DevEnv-661495','python3 -'],input=code,text=True,capture_output=True,check=True);x=json.loads(r.stdout)
(P/f'MIGRATION_{time.time_ns()}.json').write_text(json.dumps(x,indent=2));print('migrated',len(x['migrations']),'bytes',sum(j['bytes'] for j in x['migrations']),'free_GiB',x['free_GiB'])
