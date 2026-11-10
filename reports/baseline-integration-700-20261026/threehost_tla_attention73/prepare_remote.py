import pathlib,shutil,json,hashlib,subprocess
P=pathlib.Path(__file__).resolve().parent
W=pathlib.Path('/opt/tileops-all-pairs/workspace')
F=P/'frozen/TileOPs'
for name in ['benchmarks','workloads','src','scripts']:
 shutil.copytree(W/'TileOPs'/name,F/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'),dirs_exist_ok=False)
rows=[]
for f in F.rglob('*'):
 if f.is_file():rows.append({'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
(P/'frozen_hashes.json').write_text(json.dumps(rows,indent=2))
s=pathlib.Path('/opt/tileops-all-pairs/bin/allpairs-python').read_text()
assert '\\n' not in s
(P/'private-python').write_text(s)
(P/'private-python').chmod(0o755)
r=subprocess.run([str(P/'private-python'),'-c','import sys; print(sys.version); print(sys.executable)'],text=True,capture_output=True)
(P/'python_cpu_preflight.json').write_text(json.dumps(dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr),indent=2))
assert r.returncode==0
