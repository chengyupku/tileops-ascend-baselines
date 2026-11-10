"""CPU-only AST/template check; no vendor or torch imports."""
import ast,hashlib,json,pathlib,sys,types
base=pathlib.Path(__file__).parent
original=base.parent/'attention_original_build20/firstcase_adapter.py'
src=original.read_text(); tree=ast.parse(src)
f=next(x for x in tree.body if isinstance(x,ast.FunctionDef))
paged=pathlib.Path('/root/dyq/workspace/TileOPs/workloads/attention/gqa/paged.py')
pt=ast.parse(paged.read_text())
ref=next(x for x in ast.walk(pt) if isinstance(x,ast.FunctionDef) and x.name=='ref_program')
assert [x.arg for x in f.args.args]==[x.arg for x in ref.args.args][1:]
records=[]
def metadata(*args,**kwargs):
 records.append(dict(api='flash_attn_metadata',args=args,kwargs={k:str(v) for k,v in kwargs.items()}));return 'metadata-config'
def attention(*args,**kwargs):
 assert kwargs['metadata']=='metadata-config'; assert kwargs['mask_mode']==0
 records.append(dict(api='flash_attn',kwargs={k:str(v) for k,v in kwargs.items()}));return 'output','unused-lse'
m=types.ModuleType('cann_ops_transformer.ops');m.flash_attn=attention;m.flash_attn_metadata=metadata
sys.modules['cann_ops_transformer.ops']=m
ns={};exec(compile(tree,str(original),'exec'),ns)
class Shape:
 def __init__(self,shape):self.shape=shape
 def __repr__(self):return repr(self.shape)
args=[Shape((8,64,64)),Shape((256,64,8,64)),Shape((256,64,8,64)),Shape((8,32)),Shape((8,)),Shape((9,))]+[None]*5+[Shape((64,))]
assert ns['firstcase'](*args)=='output'
(base/'firstcase_adapter.py').write_text(src)
result=dict(kind='CPU AST and stub argument binding only; no numerical result',signature_order_matches_original=True,firstcase='gpt-oss-sink-decode-bfloat16-bfloat16',q_dtype='bfloat16',kv_dtype='bfloat16',sinks_dtype='float32',page_table_dtype='int32',sequence_dtype='int32',q_lengths=[1]*8,cache_lengths=[2048]*8,causal_mode0_equivalence='one last-position query per request; all valid cached keys visible',calls=records,adapter_sha256=hashlib.sha256(src.encode()).hexdigest())
(base/'CONTRACT_CHECK.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
