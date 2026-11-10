import pathlib,types,sys,json,hashlib,re
sys.dont_write_bytecode=True
W=pathlib.Path('/opt/tileops-all-pairs/workspace');P=W/'reports/baseline-integration-700-20261026/transformer_attention_misc';D=W/'TileOPs/benchmarks/ascend_all_pairs/ops_tensor';mp=D/'integration700_transformer_attention_misc_grouped.py';cp=P/'runs/ds-v3-decode-mlp-up-float16/checker.json';c=json.loads(cp.read_text())
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
for name in ['torch','numpy','benchmarks','benchmarks.ascend_baselines','benchmarks.api']:sys.modules[name]=types.ModuleType(name)
sys.modules['benchmarks.ascend_baselines'].Candidate=lambda fn,api,adapt,hashes,**kw:{'function':fn.__name__,'api':api}
sys.modules['benchmarks.ascend_baselines'].source_hashes=lambda *p:{}
sys.modules['benchmarks.api'].Implementation=lambda run:run
m=types.ModuleType('audit');m.__file__=str(mp);exec(compile(mp.read_text(),str(mp),'exec'),m.__dict__)
class ReadOnlyPath(type(pathlib.Path())):
 def mkdir(self,*a,**kw):pass
m.Path=ReadOnlyPath
ns={'PROVIDERS':{'MoEGroupedGemmFwdOp':lambda c:'prior'},'_i700_misc_grouped':m.factory}
it=(D/'__init__.py').read_text();block=it.split('# integration700 transformer_attention_misc: full original grouped host entry.',1)[1];block='\n'.join(block.strip().splitlines()[1:]);exec(compile(block,str(D/'__init__.py'),'exec'),ns)
case=types.SimpleNamespace(id=c['case_id'],params=c['case_snapshot']['parameters']);dispatch=ns['PROVIDERS']['MoEGroupedGemmFwdOp'](case);case.id='__audit_unknown__';prior=ns['PROVIDERS']['MoEGroupedGemmFwdOp'](case)
hashes={};issues=[]
for p,h in {**c['source_sha256'],**c['oracle_source_sha256']}.items():
 fp=W/p;hashes[str(fp)]=sha(fp)
 if hashes[str(fp)]!=h:issues.append('Source drift '+p)
if sha(mp)!=sha(P/mp.name):issues.append('Formal differs from tested')
root=W/'ascend_baselines/ops-tensor/examples/grouped_matmul/grouped_matmul_no_quant'
sources={str(f.relative_to(W)):sha(f) for f in root.rglob('*') if f.is_file() and f.suffix in ('.h','.cpp','.csv')}
# Traverse repository-owned quoted include closure only; never scan the SDK.
repo=W/'ascend_baselines/ops-tensor';pending=[root/'grouped_matmul_no_quant.cpp'];seen=set()
while pending:
 f=pending.pop()
 if f in seen:continue
 seen.add(f);sources[str(f.relative_to(W))]=sha(f)
 for inc in re.findall(r'^#include \"([^\"]+)\"',f.read_text(),re.M):
  for base in [f.parent,repo/'include',repo/'include/tensor_api/include',repo/'examples']:
   q=base/inc
   if q.is_file():pending.append(q);break
print(json.dumps({'key':['ops-tensor','MoEGroupedGemmFwdOp',c['case_id']],'checker_sha256':sha(cp),'module':{'path':str(mp),'sha256':sha(mp)},'provider':{'path':str(D/'__init__.py'),'sha256':sha(D/'__init__.py')},'dispatch':dispatch,'unknown':prior,'hashes':hashes,'original_sources':sources,'bridge':{'path':str(P/'grouped_bridge.cpp'),'sha256':sha(P/'grouped_bridge.cpp')},'issues':issues}))
