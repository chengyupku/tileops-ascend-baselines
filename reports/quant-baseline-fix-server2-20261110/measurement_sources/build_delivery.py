"""Export exact verified cases only; raw profiles remain outside delivery."""
import pathlib,json,hashlib,datetime,collections
P=pathlib.Path(__file__).resolve().parent
D=P/'delivery';D.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
labels={'original':'torch-eager-control','fixed':'torch-eager-layout-fixed','native':'torch-npu-antiquant','compile':'torch-compile-inductor-ascendc','compile_native':'torch-compile-inductor-antiquant','ops_math':'ops-math-native-broadcast'}
rows=json.loads((P/'RESULTS.json').read_text());tasks=json.loads((P/'TASKS.json').read_text())
registry=[];audits=[]
for x in rows:
 if x['status']!='formal_raw_verified':continue
 d=P/f"remote/runs/{x['index']}/{x['variant']}/{x['attempt']}"
 m=json.loads((d/'result.json').read_text());sources=json.loads((d/'SOURCE.json').read_text())
 matched={}
 for remote,h in sources.items():
  if remote.startswith('/opt/torch-all2208-20261105/'):
   rel=remote.removeprefix('/opt/torch-all2208-20261105/');local=P.parent/'torch-all2208-20261105'/rel
   assert local.is_file() and sha(local)==h,(local,h)
   matched[rel]=h
  elif remote.startswith('/opt/quant-baseline-fix-server2-20261110/'):
   assert sha(P/pathlib.Path(remote).name)==h
 c=json.loads((d/'correctness.json').read_text())
 assert c['inputs_preserved'] and c['check']['checked_outputs']==c['check']['total_outputs'] and not c['check']['unchecked_reason']
 libs=json.loads((d/'LOADED_LIBRARIES.json').read_text())
 stacklibs={k:v for k,v in libs.items() if '/Ascend/' in k and '/cann-' in k}
 assert stacklibs and all('/cann-9.1.0/' in k for k in stacklibs),stacklibs
 counters=m.get('dynamo_counters',{})
 if x['variant'].startswith('compile'):
  assert counters.get('stats',{}).get('unique_graphs')==1,counters
  assert not counters.get('graph_break'),counters
  assert json.loads((d/'COMPILED_ARTIFACTS.json').read_text())
 assert not any(x['aicpu_counts'])
 prep=json.loads((P/f"remote/runs/{x['index']}/prepare.json").read_text())
 registry.append({'op':x['op'],'case':x['case'],'variant':labels.get(x['variant'],x['variant']),'parameters':prep['params'],'latency_us':x['us'],'measurement':'device kernel task time;15isolated flushed windows;3warmups;1GiBflush;32AIC64AIV','kernel_task_counts':x['task_counts'],'aicpu_task_counts':x['aicpu_counts'],'graph_counters':counters,'cold_first_call_seconds':m['first_call_seconds'],'runtime_versions':m['hardware'],'compiler_threads':json.loads((d/'PROCESS_IDENTITY.json').read_text())['environment'].get('TORCHINDUCTOR_COMPILE_THREADS','runtime default'),'result_sha256':x['result_sha256'],'guard_sha256':x['guard_sha256'],'source_manifest_sha256':sha(d/'SOURCE.json'),'payload_sha256':prep['payload_sha256'],'correctness_sha256':sha(d/'correctness.json'),'attempt':x['attempt']})
 audits.append({'index':x['index'],'variant':x['variant'],'attempt':x['attempt'],'original_sources_match':matched,'loaded_cann_libraries':stacklibs,'process_identity_sha256':sha(d/'PROCESS_IDENTITY.json'),'checks_passed':True})
reference_sources={}
for audit in audits:
 for name,h in audit['original_sources_match'].items():
  assert name not in reference_sources or reference_sources[name]==h
  reference_sources[name]=h
for relative in reference_sources:
 source=P.parent/'torch-all2208-20261105'/relative
 target=D/'reference_snapshot'/relative
 target.parent.mkdir(parents=True,exist_ok=True)
 target.write_bytes(source.read_bytes())
put(D/'reference_source_hashes.json',reference_sources)
put(P/'FORMAL_SOURCE_LIBRARY_AUDIT.json',audits)
put(D/'verified_cases.json',registry)
cache=json.loads((P/'ACTUAL_COMPILED_CACHE_AUDIT.json').read_text())
assert cache['compiled_variants']==sum(x['variant'].startswith('compile') for x in rows if x['status']=='formal_raw_verified')
assert cache['loaded_binary_hashes_match'] and cache['host_compile_commands_use_cann91']
put(D/'compiled_artifact_audit.json',{k:cache[k] for k in ('collected_at','compiled_variants','exact_loaded_binary_count','cache_directories','loaded_binary_hashes_match','host_compile_commands_use_cann91','archive_sha256','scope')})
comparison=json.loads((P/'COMPARISON.json').read_text())
put(D/'comparison.json',comparison)
put(D/'attempt_outcomes.json',[{k:v for k,v in x.items() if k in ('index','op','case','variant','attempt','status','phase','error','audit_error','result_sha256')} for x in rows if x['status']!='formal_raw_verified'])
assert sha(D/'quant_baselines/layouts.py')==sha(P/'adapters.py')
assert sha(D/'quant_baselines/native.py')==sha(P/'native_adapters.py')
files=[{'file':str(f.relative_to(D)),'sha256':sha(f)} for f in sorted((D/'quant_baselines').glob('*.py'))]
put(D/'PACKAGE_STATUS.json',{'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':('verified_exact_cases_with_explicit_exclusions' if not any(r['status']=='pending' for r in json.loads((P/'COVERAGE.json').read_text())) else 'verified_exact_cases_only_campaign_in_progress'),'files':files,'formal_variant_count':len(registry),'formal_workloads':len({(r['op'],r['case']) for r in registry}),'total_scope_workloads':len(tasks),'original_reference_and_checker_unchanged':True,'raw_archives_included':False,'published':False})
print('DELIVERY_EXACT_VERIFIED',len(registry))
