"""CPU-only evidence packaging; invoke only after this task's device guard exits."""
import pathlib,json,hashlib,shutil,os,time
W=pathlib.Path('/opt/tileops-all-pairs/workspace');P=W/'reports/baseline-integration-700-20261026/threehost_tilelang_bmm_native3';records={}
def add(f,copy=False):
 f=pathlib.Path(f)
 if not f.is_file():return
 st=f.stat();r=dict(sha256=hashlib.sha256(f.read_bytes()).hexdigest(),size=st.st_size,inode=st.st_ino,device=st.st_dev);records[str(f)]=r
 if copy:
  d=P/'source_freeze'/r['sha256'];d.parent.mkdir(exist_ok=True)
  if not d.exists():shutil.copyfile(f,d)
for f in (P/'runs').glob('*/checker.json'):
 c=json.loads(f.read_text())
 for src,expected in c.get('source_sha256',{}).items():
  path=pathlib.Path(src)
  if not path.is_absolute():path=W/src
  if path.is_file():
   add(path,copy=path.suffix=='.py' and '/examples/' in str(path));records[str(path)]['checker_expected_sha256']=expected;records[str(path)]['matches_checker']=records[str(path)]['sha256']==expected
 for nf in f.parent.glob('native_load*json'):
  d=json.loads(nf.read_text());artifact=f.parent/'native_artifacts'/d['sha256'];add(artifact);assert records[str(artifact)]['sha256']==d['sha256']
sdk=W/'.cache/all-pairs/remote_native_deepgemm_deepep/sdk-920-release-headers'
add(sdk/'dependency_receipt.json',True);add(sdk/'tools/bisheng_compiler/bin/bisheng')
for f in (sdk/'asc/include').rglob('*.h'):add(f)
for src in [W/'tilelang/src/ascend/runtime/ascend_module.cc',W/'tilelang/src/ascend/stubs/ascendcl.cc',W/'tilelang/examples/ascend/example_gemm.py']:add(src,True)
for cache in ['cache/TILELANG_CACHE_DIR','tla_cache/TMPDIR']:
 for f in (P/cache).rglob('*'):
  if f.is_file():add(f)
(P/'DEPENDENCY_ARTIFACT_HASHES.json').write_text(json.dumps(dict(at=time.time(),files=records),indent=2));print('files',len(records),'checker mismatches',[p for p,r in records.items() if r.get('matches_checker') is False])
