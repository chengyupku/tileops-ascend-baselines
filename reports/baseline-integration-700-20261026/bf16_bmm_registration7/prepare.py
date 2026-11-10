import ast,hashlib,json,pathlib,pprint
O=pathlib.Path(__file__).resolve().parent; N=O.parent/'bf16_bmm_conv_grouped_native10'; G=O.parent/'bf16_gemm_registration18'; D=O.parent/'bf16_local_gemm18'; W=pathlib.Path('/root/dyq/workspace')
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(n,d):(O/n).write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
base=(G/'integration700_bf16_gemm_registration18.py').read_text(); constants={}
for node in ast.parse(base).body:
 if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name):
  try:constants[node.targets[0].id]=ast.literal_eval(node.value)
  except (ValueError,TypeError):pass
pins=json.loads((G/'FROZEN_PINS.json').read_text());excluded={}
for p,h in json.loads((N/'FROZEN.json').read_text())['sha256'].items():
 if p.endswith('/ops_transformer/__init__.py'):
  excluded[p]={'historical_sha256':h,'reason':'Shared publication routing only; historical receipts remain unchanged'};continue
 if p in pins:assert pins[p]==h,p
 pins[p]=h
cases={};indices={};envelopes={};evidence=[]
for row in json.loads((O/'SCOPE.json').read_text()):
 k=tuple(row['key']);r=pathlib.Path(row['checker']);assert sha(r)==row['checker_sha256'];receipt=json.loads(r.read_text());ad=json.loads((r.parent/'RESOURCE_ADMISSION.json').read_text());ex=json.loads((r.parent/'EXECUTION.json').read_text());cap=json.loads((r.parent/'CAPTURE.json').read_text())
 assert ex['checker_exit_code']==0 and ex['frozen_unchanged'];assert receipt['input_seed']==row['seed']
 for name,v in cap.items():
  if name.startswith('reference_0_input'):
   tail=name[len('reference_0_'):];assert v['sha256']==cap['candidate_0_'+tail]['sha256']==cap['reference_0_post_'+tail]['sha256']==cap['candidate_0_post_'+tail]['sha256']
 cases[k]=json.dumps(receipt['case_snapshot']['parameters'],sort_keys=True,separators=(',',':'));indices[k]=json.dumps(ad['params'],sort_keys=True,separators=(',',':'))
 envelopes[k[2]]={x:ad[x] for x in ('host_required','device_required','disk_required','elements_including_full_batch_weights_outputs_packing','cost')}
 for name in ('representative.json','EXECUTION.json','CAPTURE.json','RESOURCE_ADMISSION.json','GUARD_RECEIPT.json','run.sh','VALIDATION.json'):
  p=r.parent/name;pins[str(p)]=sha(p)
 evidence.append({'key':k,'receipt':str(r),'sha256':sha(r),'seed':row['seed']})
assert len(cases)==7
local=constants['LOCAL_FILES']+[str(O.parent/'bf16_bmm_conv_grouped_prep10/adapter.py'),str(W/'TileOPs/benchmarks/api.py')]
for p in local+[str(N/'FROZEN.json'),str(N/'PARENT_PARTIAL_REVIEW.json'),str(N/'capture.py'),str(N/'resource_gate.py'),str(O/'SCOPE.json')]:pins[p]=sha(p)
for p in (W/'TileOPs/src/tileops/manifest').rglob('*'):
 if p.is_file() and p.suffix in ('.py','.yaml'):pins[str(p)]=sha(p)
# Exclude only mutable shared publication router. No historical evidence rewritten.
for p in list(pins):
 if p.endswith('/ops_transformer/__init__.py'):excluded.setdefault(p,{'historical_sha256':pins[p],'reason':'Shared publication routing'});del pins[p]
dump('FROZEN_PINS.json',pins);dump('PIN_EXCLUSIONS.json',excluded)
head='"""CPU-lazy exact seven reviewed Bmm/Grouped routes; pending parent publication."""\nimport builtins, hashlib, json, os, types, tempfile, shutil, sys, ast\nfrom pathlib import Path\nfrom types import MappingProxyType\n'
for name,val in [('ROOT',str(D)),('MANIFEST',str(O/'FROZEN_PINS.json')),('MANIFEST_HASH',sha(O/'FROZEN_PINS.json')),('LOCAL_FILES',local),('CASE_PARAMS',cases),('CASE_INDICES',indices),('ENVELOPES',envelopes),('ENV',constants['ENV']),('LOADED_VENDOR',constants['LOADED_VENDOR']),('REQUIRED_LOADED',constants['REQUIRED_LOADED'])]:head+=name+' = '+pprint.pformat(val,width=120)+'\n'
helpers=base[base.index('ROOT = Path(ROOT)'):base.index('def make(case,')]
start=helpers.index('def matches(');end=helpers.index('\ndef _proc(')
helpers=helpers[:start]+(O/'private_body.txt').read_text()+'\n'+helpers[end:]
helpers=helpers.replace('BF16_GEMM','BF16_BMM').replace('bf16-gemm-registration18-','bf16-bmm-registration7-').replace('Frozen BF16 Gemm','Frozen BF16 Bmm')
# All historical evidence directories are disallowed as cache roots.
helpers=helpers.replace("if root == ROOT or ROOT in root.parents:","if any(root == p or p in root.parents for p in (ROOT, ROOT.parent/'bf16_bmm_conv_grouped_native10', ROOT.parent/'bf16_bmm_conv_grouped_prep10')):")
(O/'integration700_bf16_bmm_registration7.py').write_text(head+'\n'+helpers+'\n'+(O/'runner_body.txt').read_text())
recipe=(G/'run_environment.sh').read_text().replace('BF16_GEMM','BF16_BMM');recipe=recipe.replace('# Run in a clean shell; the wrapper validates the exact expansions.','# Require a new directory before touching caches; historical receipts are never reused.\nBF16_BMM_RUN_ROOT=$(realpath -m -- "$BF16_BMM_RUN_ROOT")\nexport BF16_BMM_RUN_ROOT\ncase "$BF16_BMM_RUN_ROOT/" in\n /root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_local_gemm18/*|/root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_bmm_conv_grouped_native10/*|/root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_bmm_conv_grouped_prep10/*)\n  echo \'Historical evidence cannot be a run root\' >&2; exit 2;;\nesac\nmkdir -m 700 -- "$BF16_BMM_RUN_ROOT"\n# Run in a clean shell; the wrapper validates the exact expansions.');(O/'run_environment.sh').write_text(recipe)
dump('CONTRACTS.json',{'status':'candidate_pending_parent_publication','accepted_added':0,'cases':[{'key':k,'params':json.loads(v),'indices':json.loads(indices[k]),'envelope':envelopes[k[2]]} for k,v in cases.items()],'evidence':evidence,'pins':len(pins),'exclusions':excluded,'implementation':'Unchanged approved adapter.candidate + private original binding.dense_fp32','admission':'constructor under live guard before checker inputs; one-use preinput token; re-admit only when no token','case_binding':'Both report params and actual resolved indices; independent deeply immutable snapshot; original Call tensor shapes preserved'})
print('generated',len(cases),'keys;',len(pins),'pins')
