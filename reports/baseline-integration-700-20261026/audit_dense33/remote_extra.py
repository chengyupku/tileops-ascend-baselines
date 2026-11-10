import sys,pathlib,json,hashlib,types,importlib,ctypes,struct,re
sys.dont_write_bytecode=True
W=pathlib.Path('/opt/tileops-all-pairs/workspace');P=W/'reports/baseline-integration-700-20261026/transformer_dense'
SCOPE=SCOPE_PLACEHOLDER
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(p),sha256=sha(p))
sys.path.insert(0,str(W/'TileOPs'));provider=importlib.import_module('benchmarks.ascend_all_pairs.ops_transformer')
# Device constructors only are stubs; provider and every formal factory execute unchanged.
for name in ['torch','benchmarks.api','benchmarks.ascend_baselines','benchmarks.ascend_math_more','workloads','workloads.rope']:
 sys.modules[name]=types.ModuleType(name)
sys.modules['torch'].int8='int8';sys.modules['torch'].__audit_stub__=True
sys.modules['benchmarks.api'].Implementation=lambda run:types.SimpleNamespace(run=run)
sys.modules['benchmarks.ascend_baselines'].Candidate=lambda impl,api,adapt,hashes,**kw:dict(run_module=impl.run.__module__,run_name=impl.run.__name__,api=api,adaptation=adapt)
sys.modules['benchmarks.ascend_baselines'].source_hashes=lambda *files:{}
sys.modules['benchmarks.ascend_baselines'].UnsupportedCase=ValueError
sys.modules['benchmarks.ascend_baselines'].WORKSPACE=W
class Fn:pass
sys.modules['benchmarks.ascend_math_more']._MathCall=lambda *a,**kw:types.SimpleNamespace(codes={},base=types.SimpleNamespace(aclCreateTensorList=Fn(),aclDestroyTensorList=Fn()))
graph=importlib.import_module('benchmarks.ascend_all_pairs.ops_transformer.integration700_transformer_dense_graph');graph.GraphCall=lambda *a:types.SimpleNamespace(cleanup=lambda:None)
sys.modules['workloads'].rope=sys.modules['workloads.rope'];sys.modules['workloads.rope'].yarn_frequency_tables=lambda:None;sys.modules['workloads.rope'].longrope_frequency_tables=lambda:None
routes=[]
for r in SCOPE:
 c=json.loads((P/r['checker_receipt']).read_text());case=types.SimpleNamespace(id=r['case_id'],params=c['case_snapshot']['parameters'],_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=r['operator'])))
 result=provider.PROVIDERS[r['operator']](case)
 mn,fn=r['factory'].replace(':','.').rsplit('.',1);assert result['run_module']==mn
 routes.append(dict(key=[r['baseline'],r['operator'],r['case_id']],factory_called=r['factory'],result=result))
assert 'torch_npu' not in sys.modules and sys.modules['torch'].__audit_stub__
# Read evidence, ELF sections without loading any shared library.
proof=json.loads((P/'YARN_MIGRATION_PROOF.json').read_text());bins=json.loads((P/'YARN_BINARY_IDENTITY.json').read_text())
def sections(path):
 b=path.read_bytes();h=struct.unpack_from('<16sHHIQQQIHHHHHH',b);off,ents,num,strings=h[6],h[11],h[12],h[13];ss=[struct.unpack_from('<IIQQQQIIQQ',b,off+i*ents) for i in range(num)];st=ss[strings];names=b[st[4]:st[4]+st[5]];out={}
 for s in ss:
  name=names[s[0]:].split(b'\0',1)[0].decode();out[name]=dict(flags=s[2],size=s[5],sha256=hashlib.sha256(b[s[4]:s[4]+s[5]]).hexdigest())
 return out
bs=[P/x['path'] for x in bins];secs=[sections(x) for x in bs];diff=[n for n in secs[0] if secs[0][n]!=secs[1][n]]
tested=(P/proof['tested_source']).read_text();registered=(P/proof['registered_source']).read_text();reconstructed=registered.replace('    try:from .integration700_transformer_dense_graph import GraphCall\n    except ImportError:from integration700_transformer_dense_graph import GraphCall\n','    from integration700_transformer_dense_graph import GraphCall\n').replace("if case.workload.tensors.get('rescale_factors') is not None:","if 'rescale_factors' in case._call.specs:")
assert tested==reconstructed and sha(P/proof['tested_source'])==proof['tested_sha256']
assert all(sha(f)==b['sha256'] for f,b in zip(bs,bins));assert diff==['.strtab'],diff
origin=json.loads((P/'GRAPH_ORIGIN_EVIDENCE.json').read_text())[0];lf=P/origin['file'];line=lf.read_text(errors='replace').splitlines()[origin['line']-1];assert line.startswith(origin['evidence'])
# Per-case native launch lines constrained to each completed checker interval.
natives=[]
for r in SCOPE:
 cp=P/r['checker_receipt'];c=json.loads(cp.read_text());needle='GroupedMatmul' if r['operator'] in ['BmmFP8FwdOp','MoEGroupedGemmFP8FwdOp','GemmINT8W8A8FwdOp'] else ('MoeInitRoutingV2' if r['operator']=='MoEPrePermuteFwdOp' else 'RotaryPositionEmbedding');hits=[];kernels=set()
 start=c['started_at'][:26].replace('T','-');end=c['finished_at'][:26].replace('T','-')
 for f in (cp.parent/'ASCEND_PROCESS_LOG_PATH').rglob('*.log'):
  for no,line in enumerate(f.read_text(errors='replace').splitlines(),1):
   if needle.lower() not in line.lower() or not any(x in line for x in ['StreamLaunchKernelV','Register Kernel succ','Op dsl file path','Compile op','compile op','LogTaskDetails']):continue
   tm=re.search(r'2026-10-26-\d\d:\d\d:\d\d\.\d+(?:\.\d+)?',line)
   if tm and not start<=tm[0].replace('.', '',0).replace('.', '.',1)[:26]<=end:continue
   if 'StreamLaunchKernelV' in line:
    k=re.search(r'kernel_name=([^, ]+)',line)
    if k:kernels.add(k[1])
   if len(hits)<8:hits.append(dict(file=str(f),line=no,text=line,log_sha256=sha(f)))
 natives.append(dict(key=[r['baseline'],r['operator'],r['case_id']],interval=[start,end],evidence=hits,launched_kernels=sorted(kernels)))
print(json.dumps(dict(routes=routes,native_per_case=natives,yarn=dict(proof=ref(P/'YARN_MIGRATION_PROOF.json'),identity_reconstructed=True,elf_binaries=[ref(f) for f in bs],elf_sections_equal_except=diff,sections=secs,original_compile=dict(**ref(lf),line=origin['line'],text=line if False else lf.read_text(errors='replace').splitlines()[origin['line']-1])),device_dynamic_libraries_loaded=False)))
