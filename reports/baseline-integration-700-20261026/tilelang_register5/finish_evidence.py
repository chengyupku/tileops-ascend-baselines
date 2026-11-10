import pathlib,json,hashlib,subprocess,shutil,datetime
W=pathlib.Path('/opt/tileops-all-pairs/workspace');P=W/'reports/baseline-integration-700-20261026/tilelang_register5';A=P/'attempt02';O=P.parent/'tla_sdk_recovery36';cache=O/'tilelang_cache/TILELANG_CACHE_DIR/0.1.15'
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
native=[]
for m in (cache/'linux-aarch64/kernels').glob('*/manifest.json'):
 manifest=json.loads(m.read_text());entry=dict(cache_key=m.parent.name,files={})
 for name,info in manifest['files'].items():
  f=m.parent/name;assert h(f)==info['sha256'];dst=A/'native/TILELANG_CACHE_DIR/0.1.15/linux-aarch64/kernels'/m.parent.name/name;shutil.copy2(f,dst);entry['files'][name]=info
 entry['params']=json.loads((m.parent/'params.json').read_text());native.append(entry)
for d in (cache/'ascend-binaries').iterdir():
 dst=A/'native/TILELANG_CACHE_DIR/0.1.15/ascend-binaries'/d.name;dst.mkdir(parents=True,exist_ok=True)
 for f in d.iterdir():
  if f.is_file():shutil.copy2(f,dst/f.name)
 f=d/'kernel.aibin';x=subprocess.run(['readelf','-h','-s',str(f)],text=True,capture_output=True);assert x.returncode==0;(dst/'kernel.readelf.txt').write_text(x.stdout+x.stderr)
 native.append(dict(binary_key=d.name,sha256=h(f),size=f.stat().st_size,metadata=json.loads((d/'metadata.json').read_text()),elf=True))
(A/'NATIVE_EVIDENCE.json').write_text(json.dumps(native,indent=2))
# Retain the actually used launcher and environment definitions as source; never execute.
for f in [pathlib.Path('/opt/tileops-all-pairs/bin/allpairs-python'),pathlib.Path('/opt/tileops-all-pairs/control/private_python.py')]:shutil.copy2(f,A/'evidence'/f.name)
x=subprocess.run(['git','-C',str(W/'tilelang'),'status','--short','--','examples/ascend/example_per_token_cast_to_fp8.py'],text=True,capture_output=True)
y=subprocess.run(['git','-C',str(W/'tilelang'),'rev-parse','HEAD'],text=True,capture_output=True)
(A/'ORIGINAL_PROVENANCE.json').write_text(json.dumps(dict(source_sha256=h(A/'frozen/vendor/examples/ascend/example_per_token_cast_to_fp8.py'),git_status=x.stdout,git_status_rc=x.returncode,git_head=y.stdout.strip(),git_head_rc=y.returncode),indent=2))
(P/'HEARTBEAT').write_text(datetime.datetime.now(datetime.timezone.utc).isoformat()+' registration complete; synchronizing evidence; zero device executions\n')
print('native manifests and ELF verified',len(native))
