import json,pathlib,sys,os,hashlib
P=pathlib.Path(__file__).resolve().parent
sys.path[:0]=[str(P/'source/src'),str(P/'source')]
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
from tileops.manifest import load_manifest,load_adts
from tileops.manifest._plan import entry_plan
from tileops.manifest._workload import instantiate
work=json.loads((P/'INPUT_WORKLIST.json').read_text());current=json.loads((P/'CURRENT_CASES.json').read_text());assert not current['errors']
bykey={tuple(r['key']):r for r in current['rows']}
old=json.loads((P.parent/'torch-all2208-20261105/CURRENT_CASES.json').read_text())['rows'];old={tuple(r['key']):r for r in old}
mapping=json.loads((P.parent/'manifest-v010-audit-20261106/comparison.json').read_text())['correspondence_950_to_nv']
queue={};unresolved=[]
def add(row,reason,oldkey=None,source='v010'):
 key=tuple(row['key']);entry=queue.setdefault(key,{'task':row,'reasons':[],'old_keys':[],'source_contract':source})
 if reason not in entry['reasons']:entry['reasons'].append(reason)
 if oldkey and list(oldkey) not in entry['old_keys']:entry['old_keys'].append(list(oldkey))
manifest=load_manifest();adts=load_adts();plans={}
for r in work['migration']['new_coverage']:
 op=r['op'];plan=plans.setdefault(op,entry_plan(op,manifest[op],adts,resolve=False));call=instantiate(plan,r['workload'],r['dtype_case']);key=('torch-eager',op,call.case_id)
 assert key in bykey,key
 add(bykey[key],'new_coverage')
requests=[((r['library'],r['op'],r['case']),'contract_review') for r in work['migration']['existing_review_or_retest']]+[(tuple(r['key']),r['reason']) for r in work['timing']]
for key,reason in requests:
 target=(key[0],mapping.get(key[1],key[1]),key[2])
 if target in bykey:add(bykey[target],reason,key)
 elif key[1]=='GQAPrefillPagedWithKVCacheFwdOp':add(old[key],reason,key,'950_extension')
 else:unresolved.append({'old_key':key,'target_key':target,'reason':reason})
# Conditional GELU reuse rejected: target widens the whole input before slicing,
# while the old reference widens each half separately. Device paths differ.
for key,row in bykey.items():
 if key[1]=='GELUAndMulFwdOp' and row['case']['params'].get('approximate')=='tanh':
  label=key[2].replace('-prefill-tanh-','-ffn-prefill-').replace('-decode-tanh-','-ffn-decode-')
  oldkey=(key[0],'GeluTanhAndMulFwdOp',label)
  assert oldkey in old
  add(row,'conditional_reuse_rejected_cast_path_changed',oldkey)
result={'target_commit':work['migration']['target_commit'],'tasks':list(queue.values()),'unresolved':unresolved,'note':'Exact case ID locates candidate only; changed contract gets target parameters, correctness still mandatory. 950-only prefill retained explicitly, not silently dropped.'}
(P/'QUEUE.json').write_text(json.dumps(result,indent=2));print(json.dumps({'tasks':len(queue),'unresolved':unresolved},indent=2))
