import sys,pathlib,json,hashlib,importlib,types,re,struct
sys.dont_write_bytecode=True
W=pathlib.Path('/opt/tileops-all-pairs/workspace');P=W/'reports/baseline-integration-700-20261026/transformer_dense';D=W/'TileOPs/benchmarks/ascend_all_pairs/ops_transformer'
SCOPE=json.loads(sys.stdin.readline()) if False else SCOPE_PLACEHOLDER
cache={}
def sha(p):
 p=pathlib.Path(p)
 if str(p) not in cache:cache[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 return cache[str(p)]
def ref(p):return dict(path=str(p),sha256=sha(p))
def resolve(p):
 p=pathlib.Path(p)
 if p.is_absolute():return p
 return P/p if (P/p).exists() else W/p
sys.path.insert(0,str(W/'TileOPs'))
p=importlib.import_module('benchmarks.ascend_all_pairs.ops_transformer')
provider_before=sha(p.__file__);rows=[];native_cache={};sources={}
for r in SCOPE:
 key=[r['baseline'],r['operator'],r['case_id']];cp=resolve(r['checker_receipt']);c=json.loads(cp.read_text());gp=resolve(r['guard_receipt']);ep=gp.with_suffix('.exit');issues=[]
 mn,fn=r['factory'].replace(':','.').rsplit('.',1);m=importlib.import_module(mn);mp=pathlib.Path(m.__file__);mh=sha(mp);sources[str(mp)]=mp.read_text()
 orig=getattr(m,fn);aliases={k:v for k,v in vars(p).items() if v is orig};stub=lambda *a: ('expected',key);setattr(m,fn,stub)
 for k in aliases:setattr(p,k,stub)
 case=types.SimpleNamespace(id=key[2],_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=key[1])))
 actual=p.PROVIDERS[key[1]](case)
 setattr(m,fn,orig)
 for k,v in aliases.items():setattr(p,k,v)
 assert actual==('expected',key),(key,actual)
 tested=c['source_sha256'].get('reports/baseline-integration-700-20261026/transformer_dense/'+mp.name)
 identity=dict(tested_sha256=tested,registered_sha256=mh,identical=tested==mh)
 if tested!=mh and key[1]!='YaRNFwdOp':issues.append('tested/formal adapter mismatch')
 critical={};original=[];drift=[]
 for path,h in dict(c['source_sha256'],**c['oracle_source_sha256']).items():
  relevant=path in c['oracle_source_sha256'] or path.endswith('libcust_opapi.so') or '/ascendc/grouped_matmul/' in path or '/ascendc/moe_init_routing_v2/' in path or '/ascendc/rotary_position_embedding/' in path
  if not relevant:continue
  f=resolve(path)
  if not f.exists():drift.append({'path':str(f),'missing':True});continue
  a=sha(f);critical[str(f)]=a
  if a!=h:drift.append({'path':str(f),'expected':h,'actual':a})
  if '/ascendc/' in path:
   tail=path.split('/ascendc/',1)[1];family='matmul' if tail.startswith('grouped_matmul/') else ('moe' if tail.startswith('moe_init_routing') else 'position_embedding')
   choices=[W/'ascend_baselines/ops-transformer'/family/tail,W/'ascend_baselines/ops-transformer/attention'/tail]
   origpath=next((x for x in choices if x.exists()),None)
   if origpath:original.append(dict(installed=str(f),original=str(origpath),sha256=a,matches=sha(origpath)==a))
 if drift:issues.append('critical source/runtime drift')
 nd=str(cp.parent);needle='GroupedMatmul' if key[1] in ('BmmFP8FwdOp','MoEGroupedGemmFP8FwdOp','GemmINT8W8A8FwdOp') else ('MoeInitRoutingV2' if key[1]=='MoEPrePermuteFwdOp' else 'RotaryPositionEmbedding')
 nk=nd+needle
 if nk not in native_cache:
  lines=[];kernels={};logs={}
  for lf in (cp.parent/'ASCEND_PROCESS_LOG_PATH').rglob('*.log'):
   for no,line in enumerate(lf.read_text(errors='replace').splitlines(),1):
    if needle.lower() not in line.lower():continue
    if any(t in line for t in ['StreamLaunchKernelV','Register Kernel succ','Op dsl file path','Compile op','compile op']):
     logs[str(lf)]=sha(lf)
     if len(lines)<35:lines.append(dict(file=str(lf),line=no,text=line,log_sha256=sha(lf)))
    for path in re.findall(r'(/[^\s\[\]\";]+\.(?:o|json))',line):
     f=pathlib.Path(path)
     if needle.lower() in path.lower() and f.is_file():kernels[path]=sha(f)
  native_cache[nk]=dict(native_evidence=lines,kernel_hashes=kernels,logs=logs)
 guardtext=gp.read_text();exitcode=int(ep.read_text().strip())
 if exitcode==76 or any(t in guardtext.lower() for t in ['interference detected','external npu process','terminating process group']):issues.append('guard contaminated')
 if exitcode and key[1]!='YaRNFwdOp':issues.append('nonzero guard')
 if c['status']!='verified' or not c['inputs_preserved'] or c['oracle_source_changed']:issues.append('checker not complete')
 if c['verification']['checked_outputs']!=c['verification']['total_outputs']:issues.append('partial outputs')
 expected_seed=int.from_bytes(hashlib.sha256(f"{key[1]}::{key[2]}".encode()).digest()[:4],'big')
 if c['input_seed']!=expected_seed:issues.append('seed mismatch')
 snapshot=cp.parent/'guard_snapshot.json'
 rows.append(dict(key=key,checker=ref(cp),guard=dict(**ref(gp),returncode=exitcode,exit=ref(ep),snapshot=ref(snapshot) if snapshot.exists() else {}),formal_module=ref(mp),formal_factory=r['factory'],tested_registered_identity=identity,host_cpu_route=dict(actual=actual,exact_route=True,aliases=list(aliases)),source_provenance=dict(primary_hashes=critical,original_comparisons=original,drift=drift,**native_cache[nk]),seed=c['input_seed'],verification=c['verification'],case_snapshot=c['case_snapshot'],observed_calls=c.get('observed_calls'),oracle_source_sha256=c['oracle_source_sha256'],checker_status=c['status'],provider_source_changed=c['provider_source_changed'],inputs_preserved=c['inputs_preserved'],oracle_source_changed=c['oracle_source_changed'],started_at=c['started_at'],finished_at=c['finished_at'],issues=issues))
assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
assert hashlib.sha256(pathlib.Path(p.__file__).read_bytes()).hexdigest()==provider_before
print(json.dumps(dict(rows=rows,provider=ref(p.__file__),formal_sources=sources,device_imported=False)))
