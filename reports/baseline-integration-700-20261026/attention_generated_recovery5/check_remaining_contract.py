import ast,pathlib,inspect,sys,types,json,hashlib
r=pathlib.Path(__file__).parent
source=r.parent/'attention_native_build5/attempt03/stage/source/attention/flash_attn/torch_extension/flash_attn.py'
tree=ast.parse(source.read_text());signatures={}
for node in tree.body:
 if isinstance(node,ast.FunctionDef) and node.name in ('flash_attn','flash_attn_metadata'):
  ps=[];required=len(node.args.args)-len(node.args.defaults)
  for i,a in enumerate(node.args.args):ps.append(inspect.Parameter(a.arg,inspect.Parameter.POSITIONAL_OR_KEYWORD,default=inspect.Parameter.empty if i<required else None))
  signatures[node.name]=inspect.Signature(ps)
calls=[]
def api(name):
 def invoke(*a,**k):
  signatures[name].bind(*a,**k);calls.append({'api':name,'kwargs':list(k),'mask_mode':k.get('mask_mode'),'win_left':k.get('win_left'),'win_right':k.get('win_right'),'layout_kv':k.get('layout_kv')});return ('out','lse') if name=='flash_attn' else 'metadata'
 return invoke
class Mask:
 def to(self,**kwargs):assert kwargs=={'dtype':'int8','device':'npu'};return self
fake=types.ModuleType('torch');fake.int8='int8';fake.ones=lambda *a:Mask();fake.triu=lambda a,diagonal:a
sys.modules['torch']=fake
pkg=types.ModuleType('cann_ops_transformer');ops=types.ModuleType('cann_ops_transformer.ops');ops.flash_attn=api('flash_attn');ops.flash_attn_metadata=api('flash_attn_metadata');sys.modules[pkg.__name__]=pkg;sys.modules[ops.__name__]=ops
from remaining_adapter import SPECS,build_run
for (op,key),s in SPECS.items():
 c=types.SimpleNamespace(id=key,_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=op)),params={'is_causal':True,'window_size_left':s['left'],'window_size_right':-1})
 f=build_run(c);args=['q','k','v','block','lengths','cuq']+[None]*5+['sinks'] if op=='GQAPagedFwdOp' else ['q','k','v','cuq','cukv']+[None]*5+['sinks'];assert f(*args)=='out'
(r/'REMAINING_CONTRACT_CHECK.json').write_text(json.dumps({'kind':'CPU AST signature binding only; no runtime/numeric claim','all_four_bind':True,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'calls':calls},indent=2))
print('All four original API bindings passed')
