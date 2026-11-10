"""CPU-only content snapshot and pre/post validation; no device imports."""
import pathlib,json,hashlib,os,sys,datetime,shutil
P=pathlib.Path(__file__).resolve().parent

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def identity(p):
 st=p.stat();return dict(realpath=str(p.resolve()),inode=st.st_ino,device=st.st_dev,size=st.st_size,mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns,sha256=sha(p))
mode=sys.argv[1];started=now()
if mode=='prepare':
 old=P.parent/'threehost_tla_other54_r2'
 shutil.copytree(old/'frozen',P/'frozen',dirs_exist_ok=True)
 pins=json.loads((P/'PINS.json').read_text());sdk=pathlib.Path(pins['environment']['ASCEND_HOME_PATH'])
 paths=set(pins['runtime'])|set(json.loads((P/'RUNTIME_PATHS.json').read_text()))|set(pins['sources'])
 roots=[sdk/'asc/include',sdk/'include',sdk/'tools/bisheng_compiler/include',sdk/'tools/bisheng_compiler/lib/clang']
 for root in roots:
  if root.exists():paths.update(str(p) for p in root.rglob('*') if p.is_file())
 for n in ['bisheng','clang','clang++','ld.lld','lld']:
  p=sdk/'tools/bisheng_compiler/bin'/n
  if p.is_file():paths.add(str(p))
 for p in (P/'frozen').rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts:paths.add(str(p))
 records={};blobs=P/'frozen_content';blobs.mkdir(exist_ok=True)
 for name in sorted(paths):
  p=pathlib.Path(name)
  if not p.is_file():raise RuntimeError('Missing '+name)
  r=identity(p);dest=blobs/r['sha256']
  if not dest.exists():shutil.copyfile(p,dest);dest.chmod(0o444)
  assert sha(dest)==r['sha256'];assert identity(p)==r
  records[name]=r
 (P/'FREEZE.json').write_text(json.dumps(dict(started=started,finished=now(),header_roots=list(map(str,roots)),files=records),indent=2))
 print('frozen',len(records),'files')
else:
 freeze=json.loads((P/'FREEZE.json').read_text());changed=[];records={}
 for name,expected in freeze['files'].items():
  current=identity(pathlib.Path(name));records[name]=current
  if current!=expected:changed.append(name)
 result=dict(started=started,finished=now(),changed=changed,files=records)
 (P/(mode+'.json')).write_text(json.dumps(result,indent=2));print(mode,'checked',len(records),'changed',len(changed));assert not changed
