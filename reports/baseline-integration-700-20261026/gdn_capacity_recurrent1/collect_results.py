import subprocess,json
from pathlib import Path
W=Path(__file__).resolve().parent
code='''import json\nfrom pathlib import Path\nw=Path(%r);d=w/'q35-gva-decode-b512-bfloat16';out={}\nfor p in [w/'HEARTBEAT.json',*d.glob('*.json'),*d.glob('*.log'),d/'loaded-maps.txt']:\n if p.is_file():out[str(p.relative_to(w))]=p.read_text()\nprint(json.dumps(out))'''%str(W)
p=subprocess.run(['/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh','openlibing-DevEnv-055409','python3 -S -B -'],input=code,capture_output=True,text=True,check=True)
files=json.loads(p.stdout)
for n,s in files.items():(W/n).write_text(s)
d=W/'q35-gva-decode-b512-bfloat16'
print('collected',list(files))
for n in ['GUARD_RECEIPT.json','checker.json']:
 if (d/n).exists():
  x=json.loads((d/n).read_text());print(n,json.dumps(x)[:2200])
print((d/'guard.stderr.log').read_text()[-1500:] if (d/'guard.stderr.log').exists() else '')
