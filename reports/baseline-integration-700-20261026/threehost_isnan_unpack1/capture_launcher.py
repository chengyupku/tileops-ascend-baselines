from pathlib import Path
import subprocess,json
P=Path(__file__).resolve().parent;W=P.parents[2];ssh=str(W/'.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh')
code="""from pathlib import Path
import json,hashlib
p=Path('/opt/tileops-all-pairs/bin/allpairs-python')
print(json.dumps(dict(path=str(p),resolved=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),content=p.read_text(),support={str(q):dict(sha256=hashlib.sha256(q.read_bytes()).hexdigest(),content=q.read_text() if q.suffix=='.py' else None) for q in [Path('/opt/tileops-all-pairs/control/private_python.py'),Path('/opt/tileops-ci/runtime/bin/python').resolve()]})))
"""
r=subprocess.run([ssh,'openlibing-DevEnv-055409','python3 -'],input=code.encode(),capture_output=True);assert r.returncode==0,r.stderr;(P/'launcher.json').write_bytes(r.stdout);print(json.loads(r.stdout)['content'][:3000])
