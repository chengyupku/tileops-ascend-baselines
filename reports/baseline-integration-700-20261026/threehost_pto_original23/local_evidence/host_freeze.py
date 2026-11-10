from pathlib import Path
import json,hashlib,platform,subprocess,sys
P=Path(__file__).resolve().parent;W=Path('/root/dyq/workspace');C=W/'.cache/all-pairs/full_rollout_20261024/repairs/resume_20261025/boundary_1557/tla_pristine_build';V=C/'source';S=Path('/usr/local/Ascend/cann-9.2.0-beta.2')
def sha(f):
 with f.open('rb') as h:return hashlib.file_digest(h,'sha256').hexdigest()
files={}
roots=[P/'runtime',P/'original',P/'frozen',P/'private_tla',V/'build',V/'3rdparty/pto-isa/include',V/'src',S/'include',S/'pkg_inc',S/'tools/bisheng_compiler/bin',S/'tools/bisheng_compiler/lib/clang']
for root in roots:
 assert root.exists(),str(root)
 for f in (list(root.glob('*.so'))+list((root/'tvm').glob('*.so')) if root==V/'build' else root.rglob('*')):
  if f.is_file() and '.cycache' not in f.parts and '__pycache__' not in f.parts:
   st=f.stat();files[str(f)]=dict(sha256=sha(f),realpath=str(f.resolve()),inode=st.st_ino,device=st.st_dev,size=st.st_size)
for f in P.glob('*.py'):files[str(f)]=dict(sha256=sha(f))
r=dict(host=platform.node(),python=sys.version,files=files,hardware=subprocess.run(['npu-guard','status','--json'],capture_output=True,text=True).stdout)
(P/'HOST_FREEZE.json').write_text(json.dumps(r,indent=2));print(json.dumps(dict(host=r['host'],count=len(files))))
