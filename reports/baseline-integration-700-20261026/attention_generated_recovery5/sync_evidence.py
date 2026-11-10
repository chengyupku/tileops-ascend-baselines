from pathlib import Path
import subprocess,tarfile,io,json,time
r=Path(__file__).resolve().parent
ssh='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
script='''import pathlib,tarfile,sys
r=pathlib.Path('/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_generated_recovery5')
paths=[p for p in r.iterdir() if p.is_file() and p.suffix in ('.json','.log','.sh','.py','.exit','.maps')]
paths+=list(r.glob('checkpoints/*/manifest.json'))+list(r.glob('checkpoint_make_receipts/*.json'))
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as t:
 for p in sorted(paths):t.add(p,arcname=str(p.relative_to(r)),recursive=False)
'''
t=time.time();p=subprocess.run([ssh,'openlibing-DevEnv-055409','python3 -'],input=script.encode(),capture_output=True);assert p.returncode==0,p.stderr.decode()
(r/'remote_evidence.tar.gz').write_bytes(p.stdout)
dest=r/'remote_evidence';dest.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(p.stdout),mode='r:gz') as tar:
 for m in tar.getmembers():
  assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
 tar.extractall(dest,filter='data')
(r/'SYNC_RECEIPT.json').write_text(json.dumps({'exit_code':p.returncode,'bytes':len(p.stdout),'elapsed':time.time()-t,'native_binaries_copied':False},indent=2));print('synced',len(p.stdout),'bytes')
