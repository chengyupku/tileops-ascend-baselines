from pathlib import Path
import json,hashlib,platform,subprocess,sys
P=Path(__file__).resolve().parent;W=P.parents[2];V=W/'ascend_baselines/tilelang-ascend';C=W/'.cache/all-pairs/remote_tla_compiler';S=W/'.cache/all-pairs/remote_native_deepgemm_deepep/sdk-920b2'
def sha(f):
 with f.open('rb') as h:return hashlib.file_digest(h,'sha256').hexdigest()
files={}
roots=[P/'runtime',P/'original',P/'frozen',P/'private_tla',C/'handoff/lib',V/'3rdparty/pto-isa/include',V/'src',S/'include',S/'pkg_inc',S/'tools/bisheng_compiler/bin',S/'tools/bisheng_compiler/lib/clang']
for root in roots:
 assert root.exists(),str(root)
 for f in root.rglob('*'):
  if f.is_file() and '.cycache' not in f.parts and '__pycache__' not in f.parts:
   st=f.stat();files[str(f)]=dict(sha256=sha(f),realpath=str(f.resolve()),inode=st.st_ino,device=st.st_dev,size=st.st_size)
for f in P.glob('*.py'):files[str(f)]=dict(sha256=sha(f))
r=dict(host=platform.node(),python=sys.version,files=files,hardware=subprocess.run(['npu-smi','info'],capture_output=True,text=True).stdout)
(P/'HOST_FREEZE.json').write_text(json.dumps(r,indent=2));print(json.dumps(dict(host=r['host'],count=len(files))))
