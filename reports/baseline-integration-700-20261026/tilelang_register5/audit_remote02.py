import pathlib,json,hashlib,shutil,subprocess
W=pathlib.Path('/opt/tileops-all-pairs/workspace');P=W/'reports/baseline-integration-700-20261026/tilelang_register5';O=P.parent/'tla_sdk_recovery36'
P.mkdir(exist_ok=True);A=P/'attempt02';A.mkdir(exist_ok=False)
def h(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def copy(p,rel):
 dst=A/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst);return dst
scope=json.loads((P/'SCOPE.json').read_text()); rows=[];pins={}
for row in scope:
 cid=row['case_id'];d=O/'runs'/('tilelang_FP8QuantFwdOp_'+cid)
 for f in d.iterdir():
  if f.is_file():copy(f,'evidence/runs/'+d.name+'/'+f.name)
 c=json.loads((d/'checker.json').read_text());g=json.loads((d/'guard.json').read_text());running=json.loads((d/'GUARD_RUNNING.json').read_text());launch=json.loads((d/'launch.json').read_text())
 assert c['status']=='verified' and c['verification']['checked_outputs']==c['verification']['total_outputs']==2 and c['inputs_preserved'] and not c['oracle_source_changed'] and not c['provider_source_changed']
 assert g['returncode']==0 and running['phase']=='running' and running['owner']=='baseline700-20261026/tla_sdk_recovery36'
 assert len(launch)>=2 and all(x['returned'] and 'TVMFFIKernelAdapter' in x['adapter_type'] for x in launch)
 mismatches=[]
 for path,digest in c['source_sha256'].items():
  f=pathlib.Path(path);f=f if f.is_absolute() else W/f
  actual=h(f)
  if actual!=digest:mismatches.append(str(f))
  pins[str(f)]=digest
 assert not mismatches,mismatches
 for path,digest in c['oracle_source_sha256'].items():assert h(W/path)==digest
 assert h(W/'TileOPs/scripts/ascend_pair_check.py')==c['verifier_sha256']
 rows.append(dict(case_id=cid,verification=c['verification'],seed=c['input_seed'],snapshot=c['case_snapshot'],edge_check=c['edge_check'],source_hash_count=len(c['source_sha256']),guard_running=running,launch=launch))
for f in ['representative.guard.json','representative.guard.log','FP8QuantFwdOp.guard.json','FP8QuantFwdOp.guard.log','run_tilelang.py','launch_tilelang.py','compile_recovered.txt','compile_recovered.py','registration.patch','DELIVERY.json']:
 if (O/f).exists():copy(O/f,'evidence/'+f)
for f in ['representative.guard.json','FP8QuantFwdOp.guard.json']:assert json.loads((O/f).read_text())['returncode']==0
impl=O/'integration700_tla_sdk_recovery36_impl.py';original=W/'tilelang/examples/ascend/example_per_token_cast_to_fp8.py'
assert h(impl)==pins[str(W/'reports/baseline-integration-700-20261026/tla_simple_tilelang/tilelang_adapter.py')]
copy(impl,'frozen/impl.py');copy(original,'frozen/vendor/examples/ascend/example_per_token_cast_to_fp8.py')
copy(W/'TileOPs/scripts/ascend_pair_check.py','evidence/ascend_pair_check.py')
for path in c['oracle_source_sha256']:copy(W/path,'evidence/'+path)
# Preserve key existing generated code and ELF evidence, not entire cache/SDK.
cache=O/'tilelang_cache'
artifacts=[]
for f in cache.rglob('*'):
 if f.is_file() and f.suffix in ['.so','.asc','.json'] and ('kernel' in f.name or f.suffix=='.json'):
  artifacts.append(f)
(A/'cache_inventory.json').write_text(json.dumps([dict(path=str(f),size=f.stat().st_size,sha256=h(f)) for f in artifacts],indent=2))
for f in artifacts:
 if f.suffix=='.asc' and 'fp8' in f.read_text(errors='replace').lower():
  copy(f,'native/'+str(f.relative_to(cache)))
  for sibling in f.parent.iterdir():
   if sibling.is_file() and sibling.suffix in ['.so','.json','.py']:
    copy(sibling,'native/'+str(sibling.relative_to(cache)))
    if sibling.suffix=='.so':
     report=subprocess.run(['readelf','-h','-s',str(sibling)],text=True,capture_output=True)
     (A/'native'/str(sibling.relative_to(cache))).with_suffix('.readelf.txt').write_text(report.stdout+report.stderr)
     assert report.returncode==0
sdk=W/'.cache/all-pairs/remote_native_deepgemm_deepep/sdk-920b2'
for f in [sdk/'tools/bisheng_compiler/bin/bisheng',sdk/'aarch64-linux/include/ascendc/asc_devkit_version.h']:
 pins[str(f)]=h(f)
pins[str(A/'frozen/impl.py')]=h(impl);pins[str(A/'frozen/vendor/examples/ascend/example_per_token_cast_to_fp8.py')]=h(original)
# Original adapter and original example are superseded only by equal-byte frozen paths.
for old in [str(W/'reports/baseline-integration-700-20261026/tla_simple_tilelang/tilelang_adapter.py'),str(original)]:pins.pop(old)
(A/'pins.json').write_text(json.dumps(pins,indent=2));(A/'AUDIT.json').write_text(json.dumps(dict(rows=rows,all_tested_hashes_match=True,byte_identical_migration=True,impl_sha256=h(impl),original_sha256=h(original),native_artifacts=len(artifacts)),indent=2))
print(json.dumps(dict(audited=len(rows),pins=len(pins),artifacts=len(artifacts))))
