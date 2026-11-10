#!/usr/bin/env python3
"""CPU-only static inventory and byte-preserving source delta; never imports runtime packages."""
import pathlib, re, json, hashlib, subprocess, datetime, collections, shutil
B=pathlib.Path(__file__).resolve().parent
R=pathlib.Path('/root/dyq/workspace/ascend_baselines/ops-nn')
HOST='openlibing-DevEnv-661495'
REMOTE='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_nn11/sources/ascend_baselines/ops-nn'
def dump(n,x): (B/n).write_text(json.dumps(x,indent=2)+'\n')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def heart(stage): dump('HEARTBEAT.json',dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),stage=stage,npu_used=False))
heart('static source closure')
ops={p.parent.name:p.parent for p in R.glob('*/*/CMakeLists.txt')}
graph={}; todo=['quant_batch_matmul_v3']; missing=[]
while todo:
 n=todo.pop()
 if n in graph: continue
 if n not in ops: missing.append(n); graph[n]=[]; continue
 s=(ops[n]/'CMakeLists.txt').read_text()
 deps=[]
 for block in re.findall(r'add_(?:all_)?modules_sources\s*\((.*?)\)',s,re.S):
  m=re.search(r'\bDEPENDENCIES\s+(.*?)(?=\b[A-Z][A-Z0-9_]+\b|$)',block,re.S)
  if m: deps.extend(re.findall(r'\b[a-z][a-z0-9_\.]*\b',m[1]))
 graph[n]=sorted(set(deps)); todo.extend(deps)
selected=set()
for name in graph:
 if name in ops: selected.update(p for p in ops[name].rglob('*') if p.is_file() and not p.is_symlink())
for d in ['common','cmake','scripts','matmul/common','quant/common']:
 selected.update(p for p in (R/d).rglob('*') if p.is_file() and not p.is_symlink())
selected.update(p for p in R.iterdir() if p.is_file() and not p.is_symlink())
# Recursive quoted includes: unique repository suffix resolution, ambiguous references remain explicit.
allfiles=[p for p in R.rglob('*') if p.is_file() and not any(x in p.parts for x in ['.git','build','build_out','third_party'])]
byname=collections.defaultdict(list)
for p in allfiles: byname[p.name].append(p)
queue=list(selected); seen=set(); unresolved=[]; edges=[]
while queue:
 p=queue.pop()
 if p in seen: continue
 seen.add(p)
 if p.suffix not in ['.h','.hpp','.cpp','.cc','.c','.inc']: continue
 for inc in re.findall(r'^\s*#\s*include\s*"([^"]+)"',p.read_text(errors='replace'),re.M):
  candidates=[q for q in byname[pathlib.Path(inc).name] if q.as_posix().endswith('/'+inc)]
  direct=(p.parent/inc).resolve()
  if direct.is_relative_to(R) and direct.is_file(): candidates=[direct]
  if len(candidates)==1:
   q=candidates[0]; edges.append({'from':str(p.relative_to(R)),'include':inc,'resolved':str(q.relative_to(R))})
   if q not in selected: selected.add(q); queue.append(q)
  else: unresolved.append({'from':str(p.relative_to(R)),'include':inc,'candidates':[str(q.relative_to(R)) for q in candidates],'reason':'external_or_generated' if not candidates else 'include_search_order_not_proven'})
files=[dict(path=str(p.relative_to(R)),sha256=sha(p),size=p.stat().st_size) for p in sorted(selected)]
dump('SOURCE_CLOSURE.json',dict(operator_graph=graph,missing_operator_directories=missing,include_edges=edges,unresolved_quoted_includes=unresolved,scope='Conservative candidate source closure; all operator subtrees retained including inactive architecture/test sources; not proven minimal/build-complete.'))
remote_code='''import pathlib,json,hashlib,platform,sys
root=pathlib.Path(%r)
sdk=pathlib.Path('/usr/local/Ascend/cann-9.1.0')
paths=json.loads(sys.stdin.readline())
r={'host_architecture':platform.machine(),'source_root':str(root),'source_top_entries':sorted(p.name for p in root.iterdir()),'files':{},'sdk':{}}
for x in paths:
 p=root/x
 r['files'][x]={'exists':p.is_file(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None}
for x in ['aarch64-linux/tikcpp/ascendc_kernel_cmake','compiler/tikcpp/ascendc_kernel_cmake','include/ge/cmake/GenerateEsPackage.cmake','tools/bisheng_compiler/bin/bisheng','compiler/ccec_compiler/bin/ccec','aarch64-linux/asc','include/aclnn','lib64/libnnopbase.so']:
 p=sdk/x;r['sdk'][x]={'exists':p.exists(),'realpath':str(p.resolve())}
r['sdk_version_files']={str(p):p.read_text(errors='replace')[:20000] for p in sdk.glob('**/version.info') if len(p.parts)<10}
print(json.dumps(r))
''' % REMOTE
# stdin is JSON inventory followed by Python source? use -c quoted source, inventory stdin only.
import shlex
proc=subprocess.run(['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',HOST,'python3 -I -S -B -c '+shlex.quote(remote_code)],input=json.dumps([x['path'] for x in files])+'\n',text=True,capture_output=True,timeout=60)
if proc.returncode: raise RuntimeError(proc.stderr)
remote=json.loads(proc.stdout);dump('REMOTE_SOURCE_SDK.json',remote)
heart('copying original source delta')
for f in files:
 old=remote['files'][f['path']]
 f['remote_status']='identical' if old['sha256']==f['sha256'] else 'different' if old['exists'] else 'missing'
 f['remote_sha256']=old['sha256']
 f['bundled']=f['remote_status']!='identical'
 if f['bundled']:
  dest=B/'source_delta'/f['path'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(R/f['path'],dest)
  assert sha(dest)==f['sha256']
dump('BUNDLE_MANIFEST.json',dict(host=HOST,target_architecture='aarch64',payload_type='original architecture-neutral source only; no prebuilt libraries/runtime/OPP',source_root=str(R),remote_base=REMOTE,files=files,closure_proven=False,ready=False,counts=dict(collections.Counter(x['remote_status'] for x in files))))
heart('candidate bundle assembled; unresolved dependencies recorded')
print(json.dumps({'graph':graph,'files':len(files),'unresolved_includes':len(unresolved),'delta':sum(x['bundled'] for x in files),'remote_counts':dict(collections.Counter(x['remote_status'] for x in files))},indent=2))
