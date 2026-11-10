"""Finalize only closed campaign attempts; immutable manifest and guarded cleanup."""
import pathlib,json,hashlib,sys,tarfile,os,shutil,gzip
P=pathlib.Path(__file__).resolve().parent;mode,host,tid,attempt=sys.argv[1:];out=P/host/'runs'/tid/attempt
assert out.resolve().is_relative_to(P.resolve()) and (out/'DONE.json').exists()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
if mode=='pack':
 files={str(f.relative_to(out)):{'size':f.stat().st_size,'sha256':sha(f)} for f in out.rglob('*') if f.is_file() and f.name not in ('FILES.json','ARCHIVED.json')}
 (out/'FILES.json').write_text(json.dumps({'files':files}))
 # Streaming gzip avoids a second large remote copy.
 with gzip.GzipFile(fileobj=sys.stdout.buffer,mode='wb',compresslevel=1) as zipped:
  with tarfile.open(fileobj=zipped,mode='w|') as tar:
   for f in sorted(out.rglob('*')):
    if f.is_file():tar.add(f,arcname=str(f.relative_to(out)),recursive=False)
elif mode=='clean':
 receipt=json.load(sys.stdin);assert receipt['verified'] and receipt['remote']==str(out)
 files=json.loads((out/'FILES.json').read_text())['files']
 for name,meta in files.items():assert (out/name).stat().st_size==meta['size'] and sha(out/name)==meta['sha256']
 # Preserve full local archive; remove only generated task profiler data remotely.
 if (out/'profiles').exists():shutil.rmtree(out/'profiles')
 (out/'ARCHIVED.json').write_text(json.dumps(receipt))
