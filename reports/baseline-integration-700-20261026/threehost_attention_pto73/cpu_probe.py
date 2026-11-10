import pathlib,json,hashlib,sys,platform,shutil,difflib
W=pathlib.Path('/opt/tileops-all-pairs/workspace'); V=W/'ascend_baselines/tilelang-ascend'; R=W/'.cache/all-pairs/full_rollout_20261024/remote_ready/server1/group1/recovery_resume25/tla_supplemental_1858/runtime_env2/runtime/package'
def ident(p):
 if not p.is_file():return {'path':str(p),'exists':False}
 s=p.stat();return dict(path=str(p),realpath=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),inode=s.st_ino,size=s.st_size)
files=['tilelang/jit/__init__.py','tilelang/jit/adapter/libgen.py','tilelang/utils/target.py','tilelang/engine/phase.py','tilelang/engine/lower.py','examples/attention_sink/example_gqa_sink_fwd_varlen/example_gqa_sink_fwd_varlen.py','3rdparty/pto-isa/include/pto/pto-inst.hpp']
r=dict(host=platform.node(),machine=platform.machine(),python=sys.version,executable=sys.executable,stdlib=str(pathlib.__file__),wrapper=ident(pathlib.Path('/opt/tileops-all-pairs/bin/allpairs-python')),vendor={},old_package={},diffs={})
for f in files:
 r['vendor'][f]=ident(V/f);r['old_package'][f]=ident(R/f)
 if (V/f).is_file() and (R/f).is_file():r['diffs'][f]=''.join(difflib.unified_diff((R/f).read_text().splitlines(True),(V/f).read_text().splitlines(True)))
r['python_closure']={'same':0,'different':[],'missing_old':[]}
for p in (V/'tilelang').rglob('*.py'):
 rel=p.relative_to(V);q=R/rel
 if not q.is_file():r['python_closure']['missing_old'].append(str(rel))
 elif p.read_bytes()==q.read_bytes():r['python_closure']['same']+=1
 else:r['python_closure']['different'].append(str(rel))
r['compiler_files']=[ident(p) for p in (W/'.cache/all-pairs/remote_tla_compiler/handoff/lib').glob('*.so')]
r['bisheng']=shutil.which('bisheng')
print(json.dumps(r,indent=2))
