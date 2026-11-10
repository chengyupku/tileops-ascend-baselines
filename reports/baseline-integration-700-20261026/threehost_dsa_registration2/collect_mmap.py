import pathlib,mmap,hashlib,os,json,datetime
p=pathlib.Path('/usr/local/Ascend/cann-9.1.0/aarch64-linux/lib64/libacl_op_compiler.so')
with p.open('rb') as f:
 s=os.fstat(f.fileno())
 with mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as m:
  maps=[x for x in pathlib.Path('/proc/self/maps').read_text().splitlines() if str(p) in x]
  print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),method='read-only file mmap; no dlopen/native execution',sha256=hashlib.sha256(m).hexdigest(),stat_device=s.st_dev,inode=s.st_ino,bytes=len(m),maps=maps,mountinfo=pathlib.Path('/proc/self/mountinfo').read_text())))
