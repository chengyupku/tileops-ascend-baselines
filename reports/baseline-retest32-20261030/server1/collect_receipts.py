"""Incremental CPU-only receipt collection for completed server1 attempts."""
import json,subprocess
from pathlib import Path
H=Path(__file__).resolve().parent
rows=json.loads((H/'journal.snapshot.json').read_text())
for row in rows:
    folder=H/'receipts'/row['id']/Path(row['measurement']).parent.name
    if (folder/'collected.json').exists():continue
    paths=[row['measurement'],row['guard']]
    script='''import json\nfrom pathlib import Path\npaths='''+repr(paths)+'''\nm=Path(paths[0]); files=[m,Path(paths[1]),m.parent/'correctness.json',m.parent/'prepare.json',m.parent/'launch_environment.json']\nprint(json.dumps({p.name:json.loads(p.read_text()) for p in files if p.exists()}))\n'''
    r=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ClearAllForwardings=yes','openlibing-DevEnv-661495','python3 -'],input=script,text=True,capture_output=True,check=True)
    data=json.loads(r.stdout);folder.mkdir(parents=True,exist_ok=True)
    for name,value in data.items():(folder/name).write_text(json.dumps(value,ensure_ascii=False,indent=2))
    (folder/'collected.json').write_text(json.dumps(row,ensure_ascii=False,indent=2))
print('Receipt collection complete:',len(rows),'journal entries')
