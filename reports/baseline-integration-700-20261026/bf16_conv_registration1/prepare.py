import ast, hashlib, json, pathlib, pprint, datetime
O=pathlib.Path(__file__).resolve().parent; R=O.parent; W=R.parents[1]; G=R/'bf16_gemm_registration18'; N=R/'bf16_bmm_conv_grouped_native10'; P=R/'bf16_bmm_conv_grouped_prep10'; B=W/'TileOPs/benchmarks'
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(n,d): (O/n).write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
if (O/'FEEDBACK.txt').exists():print((O/'FEEDBACK.txt').read_text())
dump('HEARTBEAT.json',dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase='freeze and generate',accepted_added=0))
receipt=N/'conv-firstcanonical/representative.json'; rr=json.loads(receipt.read_text()); review=json.loads((N/'PARENT_CONV_REVIEW.json').read_text())
assert sha(receipt)==review['rows'][0]['checker_sha256'] and rr['status']=='verified'
k=('ops-transformer','Conv1dFwdOp','whisper-conv1-bias-bfloat16'); assert tuple(review['rows'][0]['key'])==k
row=next(r for r in json.loads((N/'CASES.json').read_text()) if tuple(r['key'])==k)
ad=json.loads((receipt.parent/'RESOURCE_ADMISSION.json').read_text()); indices=dict(ad['params']);indices.pop('bias')
assert all(ad['params'][n]==v for n,v in row['manifest_params'].items())
pins=json.loads((G/'FROZEN_PINS.json').read_text());excluded={}
for p,h in json.loads((N/'FROZEN_CONV.json').read_text())['sha256'].items():
 p=str(pathlib.Path(p).resolve())
 if p.endswith('/ops_transformer/__init__.py'):
  excluded[p]={'historical_sha256':h,'reason':'Parent-authorized publication routing; original receipts retained unchanged.'};pins.pop(p,None);continue
 if p in pins:assert pins[p]==h,p
 pins[p]=h
extra=[N/'FROZEN_CONV.json',N/'PARENT_CONV_REVIEW.json',N/'CASES.json',N/'capture_conv.py',N/'resource_gate.py',P/'adapter.py',B/'api.py',O/'SCOPE.json']
extra += [receipt.parent/n for n in ('representative.json','CAPTURE.json','EXECUTION.json','RESOURCE_ADMISSION.json','GUARD_RECEIPT.json','NATIVE_DISPATCH.log','run.sh')]
for p in extra:pins[str(p)]=sha(p)
# Freeze actual captured bytes, without numerical interpretation.
cap=json.loads((receipt.parent/'CAPTURE.json').read_text())
for n,d in cap.items():
 if d.get('stored'):
  p=receipt.parent/(n+'.bin');assert sha(p)==d['sha256'];pins[str(p)]=d['sha256']
assert sha(P/'adapter.py')==sha(N/'adapter.py')
dump('FROZEN_PINS.json',pins);dump('PIN_EXCLUSIONS.json',excluded)
base=(G/'integration700_bf16_gemm_registration18.py').read_text();tree=ast.parse(base)
constants={}
for node in tree.body:
 if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name):
  try:constants[node.targets[0].id]=ast.literal_eval(node.value)
  except (ValueError,TypeError):pass
constants.update(MANIFEST=str(O/'FROZEN_PINS.json'),MANIFEST_HASH=sha(O/'FROZEN_PINS.json'),CASE_PARAMS={k:json.dumps(rr['case_snapshot']['parameters'],sort_keys=True,separators=(',',':'))},ENVELOPES={k[2]:row['envelope']},CASE_INDICES=indices)
constants['LOCAL_FILES']=[str(N/'resource_gate.py'),str(B/'ascend_baselines.py'),str(B/'ascend_math_more.py'),str(R/'bf16_local_gemm18/binding.py'),str(P/'adapter.py'),str(B/'api.py')]
head='"""CPU-lazy exact Conv1 registration; separate parent publication required."""\nimport builtins, hashlib, json, os, types, tempfile, shutil, sys\nfrom pathlib import Path\nfrom types import MappingProxyType\n'
for name in ('ROOT','MANIFEST','MANIFEST_HASH','LOCAL_FILES','CASE_PARAMS','CASE_INDICES','ENVELOPES','ENV','LOADED_VENDOR','REQUIRED_LOADED'):head+=name+' = '+pprint.pformat(constants[name],width=120)+'\n'
body=base[base.index('ROOT = Path(ROOT)'):]
start=body.index('def matches(');end=body.index('def _proc(')
body=body[:start]+(O/'case_and_load.txt').read_text()+'\n\n'+body[end:]
start=body.index('    frozen = types.SimpleNamespace(');end=body.index('    prepared = ',start)
body=body[:start]+'    frozen = _freeze_case(case)\n'+body[end:]
body=body.replace("        child = private['binding'].dense_fp32(frozen)\n        try:","        owned = []\n        def owned_factory(proxy):\n            child = private['binding'].dense_fp32(proxy)\n            owned.append(child)\n            return child\n        try:\n            child = private['adapter'].candidate(frozen, native_factory=owned_factory)")
body=body.replace('            if child.cleanup is not None: child.cleanup()', '            for item in reversed(owned):\n                if item.cleanup is not None: item.cleanup()')
body=body.replace('    prepared = _prepare_admission(key) if _guard_identity() is not None else None', '    prepared = _prepare_admission(key) if _guard_identity() is not None or _in_guard_group() or os.environ.get("BF16_GEMM_GUARD_OWNER") else None')
body=body.replace('    import torch, torch_npu\n    free,total = torch.npu.mem_get_info()', '    free,total = _device_capacity()')
body=body.replace('BF16_GEMM','BF16_CONV').replace('BF16 Gemm','BF16 Conv1').replace('bf16-gemm-registration18-','bf16-conv-registration1-').replace('exact eighteen canonical BF16 Gemm cases.','exact canonical BF16 Conv1 case.')
# Reject using any historic evidence subtree as writable runtime cache.
body=body.replace("    if root == ROOT or ROOT in root.parents:","    evidence = Path('/root/dyq/workspace/reports/baseline-integration-700-20261026')\n    if root == evidence or evidence in root.parents:")
(O/'integration700_bf16_conv_registration1.py').write_text(head+body)
recipe=(G/'run_environment.sh').read_text().replace('BF16_GEMM','BF16_CONV')
recipe=recipe.replace('for variable in ASCEND_CACHE_PATH', '# Reject evidence caches and reused roots before any directory creation.\npython3 -S - "$BF16_CONV_RUN_ROOT" <<\'CPU_ROOT_CHECK\'\nimport pathlib,sys\np=pathlib.Path(sys.argv[1]).resolve()\ne=pathlib.Path(\'/root/dyq/workspace/reports/baseline-integration-700-20261026\')\nif p==e or e in p.parents or p.exists():\n    raise SystemExit(\'RUN_ROOT must be fresh and outside historical evidence tree\')\np.mkdir(parents=True)\nCPU_ROOT_CHECK\nfor variable in ASCEND_CACHE_PATH')
(O/'run_environment.sh').write_text(recipe)
dump('CONTRACTS.json',dict(status='candidate_pending_parent_review_publication',accepted_added=0,key=k,case_params=rr['case_snapshot']['parameters'],indices=indices,raw_manifest_params=row['manifest_params'],envelope=row['envelope'],pin_count=len(pins),excluded=excluded,numerical_evidence={str(receipt):sha(receipt)},admission='Guard-bound pre-input constructor admission, one-use token; no second peak check after reference',algorithm='Unmodified approved adapter.candidate + original private binding.dense_fp32; original bias/packing/reference/seed/tolerance preserved',snapshot='Immutable tuple tensor shapes/dtypes; call.indices, tensors, params, present API; runtime revalidation'))
print('Generated:',len(pins),'pins;',len(excluded),'publication exclusion')
