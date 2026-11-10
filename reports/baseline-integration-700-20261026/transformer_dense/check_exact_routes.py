"""Device-free routing check; substitutes factory call targets only within this process."""
import sys,importlib,json,types
from pathlib import Path
sys.dont_write_bytecode=True
p=Path(__file__).parent;w=Path('/opt/tileops-all-pairs/workspace');sys.path[:0]=[str(w/'TileOPs'),str(w/'TileOPs/src'),str(w)]
m=importlib.import_module('benchmarks.ascend_all_pairs.ops_transformer');saved={};rows=[]
for name,value in list(vars(m).items()):
 module=getattr(value,'__module__','')
 if callable(value) and module.startswith('benchmarks.ascend_all_pairs.ops_transformer.integration700_transformer_dense_'):
  saved[name]=value;factory=module+'.'+value.__name__;setattr(m,name,lambda case,factory=factory:factory)
try:
 for row in json.loads((p/'RESULT.json').read_text())['rows']:
  if row['status']!='completed':continue
  case=types.SimpleNamespace(id=row['case_id'],_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=row['operator'])))
  actual=m.PROVIDERS[row['operator']](case);assert actual==row['factory'],(row['operator'],row['case_id'],actual,row['factory'])
  rows.append(dict(key=[row['baseline'],row['operator'],row['case_id']],factory=actual))
 assert 'torch_npu' not in sys.modules
finally:
 for name,value in saved.items():setattr(m,name,value)
(p/'EXACT_ROUTES_DEVICE_FREE.json').write_text(json.dumps(dict(rows=rows,count=len(rows),torch_npu_imported=False,numerical_verification=False,note='Static routing only; numerical evidence remains original unchanged checker receipts.'),indent=2));print('exact routes passed:',len(rows))
