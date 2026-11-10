"""Read-only collection of exact loaded AscendC cache directories; no NPU imports."""
import pathlib,json,hashlib,subprocess,tarfile,shlex,datetime,ast
P=pathlib.Path(__file__).resolve().parent
rows=[r for r in json.loads((P/'RESULTS.json').read_text()) if r['status']=='formal_raw_verified' and r['variant'].startswith('compile')]
expected={};bindings=[];dirs=set()
for r in rows:
 d=P/f"remote/runs/{r['index']}/{r['variant']}/{r['attempt']}"
 libs=json.loads((d/'LOADED_LIBRARIES.json').read_text())
 selected={k:v for k,v in libs.items() if k.startswith('/tmp/.npu_kernels_root/')}
 for k,v in selected.items():
  assert k not in expected or expected[k]==v,(k,'changed loaded binary')
  expected[k]=v
  parts=pathlib.PurePosixPath(k).parts
  dirs.add(str(pathlib.PurePosixPath(*parts[:5])))
 bindings.append({'index':r['index'],'variant':r['variant'],'attempt':r['attempt'],'loaded_libraries':selected})
archive=P/'ACTUAL_COMPILED_CACHE.private.tar.gz'
paths=[d.lstrip('/') for d in sorted(dirs)]
paths.append('opt/quant-baseline-fix-server2-20261110/cache/inductor')
with archive.open('wb') as out:
 subprocess.run(['ssh','openlibing-DevEnv-055409','tar -C / -czf - -- '+ ' '.join(shlex.quote(p) for p in paths)],stdout=out,check=True)
root=P/'actual_compiled_cache';root.mkdir(exist_ok=True)
with tarfile.open(archive) as t:t.extractall(root,filter='data')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for k,v in expected.items():assert sha(root/k.lstrip('/'))==v,k
generated={}
for r in rows:
 d=P/f"remote/runs/{r['index']}/{r['variant']}/{r['attempt']}"
 for k,v in json.loads((d/'COMPILED_ARTIFACTS.json').read_text()).items():
  f=root/k.lstrip('/')
  assert f.is_file() and sha(f)==v,k
  if f.suffix=='.py' and k not in generated:
   tree=ast.parse(f.read_text())
   calls=[]
   for node in ast.walk(tree):
    if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name=='call':
     calls.extend(ast.unparse(n.func) for n in ast.walk(node) if isinstance(n,ast.Call))
   generated[k]={'sha256':v,'runtime_calls':calls}
(P/'GENERATED_WRAPPER_REVIEW.json').write_text(json.dumps({'scope':'Cache snapshot includes earlier attempts; call lists are static inspection, not proof every file executed. Actual execution bindings are LOADED_LIBRARIES and profiler records.','files':generated},indent=2)+'\n')
commands=[]
for d in sorted(dirs):
 f=root/d.lstrip('/')/'inductor_wrapper.cpp'
 line=f.read_text().splitlines()[0]
 assert '/cann-9.1.0/include' in line and '/cann-9.1.0/lib64' in line,line
 assert 'cann-9.2' not in line
 commands.append({'directory':d,'wrapper_source_sha256':sha(f),'recorded_host_compile_command':line})
files=[{'path':str(f.relative_to(root)),'bytes':f.stat().st_size,'sha256':sha(f)} for f in sorted(root.rglob('*')) if f.is_file()]
report={'collected_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'compiled_variants':len(rows),'exact_loaded_binary_count':len(expected),'cache_directories':len(dirs),'loaded_binary_hashes_match':True,'host_compile_commands_use_cann91':True,'bindings':bindings,'host_compile_commands':commands,'files':files,'archive_sha256':sha(archive),'scope':'Post-run exact loaded binaries matched to hashes recorded in each measurement process; generated sources and recorded host commands retained. Does not invent historical compiler execution receipts. Cache is private; not exported.'}
(P/'ACTUAL_COMPILED_CACHE_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
print({k:v for k,v in report.items() if k not in ('bindings','host_compile_commands','files')})
