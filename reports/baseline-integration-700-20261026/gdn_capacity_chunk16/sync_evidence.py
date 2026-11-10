from pathlib import Path
import subprocess,shlex,tarfile,io,json
W=Path(__file__).resolve().parent
ssh='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
code="from pathlib import Path; import sys,tarfile; w=Path(sys.argv[1]); t=tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz'); files=[p for p in w.iterdir() if p.is_file() and p.suffix in ('.json','.diff')]; files += [p for d in w.iterdir() if d.is_dir() and d.name.startswith('q35-') for p in d.iterdir() if p.is_file()]; [t.add(p,arcname=str(p.relative_to(w))) for p in files]; t.close()"
p=subprocess.run([ssh,'openlibing-DevEnv-055409',shlex.join(['python3','-S','-B','-c',code,str(W)])],capture_output=True);assert p.returncode==0,p.stderr
with tarfile.open(fileobj=io.BytesIO(p.stdout),mode='r:gz') as t:t.extractall(W,filter='data')
print('synced',len(p.stdout),'bytes')
