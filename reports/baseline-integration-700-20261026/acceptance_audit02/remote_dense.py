import json,pathlib,hashlib,sys,types,re
sys.dont_write_bytecode=True
W=pathlib.Path('/opt/tileops-all-pairs/workspace');P=W/'reports/baseline-integration-700-20261026/transformer_dense';D=W/'TileOPs/benchmarks/ascend_all_pairs/ops_transformer'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
for name in ['torch','benchmarks','benchmarks.api','benchmarks.ascend_baselines','benchmarks.ascend_math_more','integration700_transformer_dense_graph']:sys.modules[name]=types.ModuleType(name)
sys.modules['torch'].int8='int8'
sys.modules['benchmarks.api'].Implementation=lambda run:run
sys.modules['benchmarks.ascend_baselines'].Candidate=lambda fn,api,adapt,hashes,**kw:{'function':fn.__name__,'api':api}
sys.modules['benchmarks.ascend_baselines'].source_hashes=lambda *p:{}
sys.modules['benchmarks.ascend_baselines'].UnsupportedCase=ValueError
class Fn:pass
sys.modules['benchmarks.ascend_math_more']._MathCall=lambda *a,**kw:types.SimpleNamespace(codes={},base=types.SimpleNamespace(aclCreateTensorList=Fn(),aclDestroyTensorList=Fn()))
sys.modules['integration700_transformer_dense_graph'].GraphCall=lambda *a:types.SimpleNamespace(cleanup=lambda:None)
rows=[];it=(D/'__init__.py').read_text()
for suffix,op,cid,rel in [('post','MoEPostPermuteFwdOp','decode-t32-float16','post_check_graph/decode-t32-float16.json'),('quant','GemmINT8W8A8FwdOp','llama3-70b-dec-qkv-tsr-bfloat16','quant_check_scale_shape/representative.json')]:
 cp=P/rel;c=json.loads(cp.read_text());mp=D/('integration700_transformer_dense_'+suffix+'.py');m=types.ModuleType('audit');m.__file__=str(mp);exec(compile(mp.read_text(),str(mp),'exec'),m.__dict__);issues=[];hashes={}
 expected=c['source_sha256']['reports/baseline-integration-700-20261026/transformer_dense/'+mp.name]
 if sha(mp)!=expected:issues.append('Formal adapter differs')
 patch=(P/('provider_init.'+suffix+'.patch')).read_text();body='\n'.join(patch.strip().splitlines()[2:]);ns={'PROVIDERS':{op:lambda c:'prior'},'_integration700_transformer_dense_'+op+'_factory':getattr(m,suffix)}
 if patch.strip() not in it:issues.append('Exact registration patch absent from live provider')
 exec(compile(body,str(D/'__init__.py'),'exec'),ns)
 case=types.SimpleNamespace(id=cid,params=c['case_snapshot']['parameters'],inputs=[types.SimpleNamespace(dtype='torch.float16')]);actual=ns['PROVIDERS'][op](case);case.id='__audit_unknown__';prior=ns['PROVIDERS'][op](case)
 selected={k:v for k,v in c['source_sha256'].items() if k.endswith('libcust_opapi.so') or ('/ascendc/'+('moe_finalize_routing_v2' if suffix=='post' else 'grouped_matmul')+'/') in k or k.endswith(mp.name)}
 selected.update(c['oracle_source_sha256']);comparisons=0; source_records=[]
 for p,h in selected.items():
  fp=pathlib.Path(p);fp=fp if fp.is_absolute() else W/fp;actualhash=sha(fp)
  if actualhash!=h:issues.append('Source/runtime drift '+p)
  if '/ascendc/' in p:
   source_records.append({'path':p,'sha256':actualhash})
   tail=p.split('/ascendc/',1)[1];orig=W/'ascend_baselines/ops-transformer'/('moe' if suffix=='post' else 'matmul')/tail
   if orig.exists():
    comparisons+=1
    if sha(orig)!=actualhash:issues.append('Installed source not original '+p)
  elif fp.suffix=='.so' or fp.name==mp.name:hashes[str(fp)]=actualhash
 # Actual native launch and source selection; do not confuse graph registration with launch.
 evidence=[];kernels={}
 needle='moefinalizeroutingv2' if suffix=='post' else 'GroupedMatmul_9fadc675eeac2d94d4d5d4cbba2c95d4'
 for f in (cp.parent/'ASCEND_PROCESS_LOG_PATH').rglob('*.log'):
  for i,line in enumerate(f.read_text(errors='replace').splitlines(),1):
   if needle.lower() not in line.lower():continue
   if 'StreamLaunchKernelV' in line or 'Register Kernel succ' in line or 'Op dsl file path' in line:
    evidence.append({'file':str(f),'line':i,'text':line,'log_sha256':sha(f)})
   for match in re.findall(r'(/[^\s\[\]\";]+\.(?:o|json))',line):
    if 'te_moefinalizerouting' in match or 'GroupedMatmul_9fadc675' in match:
     fp=pathlib.Path(match)
     if fp.is_file():kernels[str(fp)]=sha(fp)
 rows.append({'key':['ops-transformer',op,cid],'checker_sha256':sha(cp),'module':{'path':str(mp),'sha256':sha(mp)},'dispatch':actual,'unknown':prior,'issues':issues,'source_compared_count':comparisons,'source_records':source_records,'primary_hashes':hashes,'native_evidence':evidence,'kernel_hashes':kernels})
print(json.dumps({'rows':rows,'provider':{'path':str(D/'__init__.py'),'sha256':sha(D/'__init__.py')}}))
