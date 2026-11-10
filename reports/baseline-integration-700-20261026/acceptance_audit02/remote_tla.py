import json,pathlib,hashlib,sys,types,functools,re
sys.dont_write_bytecode=True
W=pathlib.Path('/opt/tileops-all-pairs/workspace');P=W/'reports/baseline-integration-700-20261026/tla_simple_tilelang';D=W/'TileOPs/benchmarks/ascend_all_pairs/tla'
keys=[('SqrtFwdOp','256M-bfloat16','sqrt7'),('SqrtFwdOp','256M-float16','sqrt7'),('BmmFwdOp','square-256-bfloat16','gemm7'),('GemmFwdOp','ds-v3-m16-mlp-up-bfloat16','gemm7'),('MoEGroupedGemmFwdOp','ds-v3-decode-down-bfloat16-float32','moe7')]
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
ns={'partial':functools.partial,'PROVIDERS':{op:lambda case:'prior' for op,_,_ in keys}};mods={};it=(D/'__init__.py').read_text()
for suffix in ['sqrt7','gemm7','moe7','gemm_rest7']:
 name='integration700_tla_simple_tilelang_'+suffix;mp=D/(name+'.py');m=types.ModuleType(name);m.__file__=str(mp);exec(compile(mp.read_text(),str(mp),'exec'),m.__dict__);mods[suffix]=m
 # _prepare is runtime boundary; actual formal factory performs key+parameter checks and dispatches.
 m._prepare=lambda suffix=suffix:(None,None,types.SimpleNamespace(factory=lambda c,op,n,t,o:(suffix,op,c.id)),None)
 alias='_i700_tla_simple_'+suffix;ns[alias]=m
 block=it.split('# BEGIN '+name+'\n',1)[1].split('# END '+name,1)[0];block='\n'.join(block.splitlines()[1:]);exec(compile(block,str(D/'__init__.py'),'exec'),ns)
rows=[]
for op,cid,suffix in keys:
 cp=P/'runs'/(op+'_'+cid+'_v7')/'checker.json';c=json.loads(cp.read_text());m=mods[suffix];bad=[];hashes={}
 for p,h in {**c['source_sha256'],**c['oracle_source_sha256'],**m.config()['pins']}.items():
  fp=pathlib.Path(p);fp=fp if fp.is_absolute() else W/fp
  if not fp.exists():bad.append('missing '+str(fp));continue
  actual=sha(fp);hashes[str(fp)]=actual
  if actual!=h:bad.append('hash mismatch '+str(fp))
 case=types.SimpleNamespace(id=cid,params=c['case_snapshot']['parameters']);dispatch=ns['PROVIDERS'][op](case);case.id='__audit_unknown__';prior=ns['PROVIDERS'][op](case)
 calls=[]
 for f in cp.parent.glob('jit/launch-evidence/*/calls-*.json'):
  x=json.loads(f.read_text());calls.append({'path':str(f),'sha256':sha(f),'successful_host_launches':x['successful_host_launches'],'pid':x['pid']})
 rows.append({'key':['tilelang-ascend',op,cid],'issues':bad,'checker_sha256':sha(cp),'hashes':hashes,'module':{'path':str(D/('integration700_tla_simple_tilelang_'+suffix+'.py')),'sha256':sha(D/('integration700_tla_simple_tilelang_'+suffix+'.py'))},'dispatch':dispatch,'unknown':prior,'calls':calls})
print(json.dumps({'rows':rows,'provider':{'path':str(D/'__init__.py'),'sha256':sha(D/'__init__.py')}}))
