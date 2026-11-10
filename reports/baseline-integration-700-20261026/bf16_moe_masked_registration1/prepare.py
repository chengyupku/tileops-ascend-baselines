import ast,hashlib,json,pathlib,pprint,sys,datetime
sys.dont_write_bytecode=True
O=pathlib.Path(__file__).resolve().parent;R=O.parent;W=R.parents[1];N=R/'bf16_moe_masked_recovery1';C=R/'bf16_conv_registration1';B=W/'TileOPs/benchmarks'
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(n,d):(O/n).write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
if (O/'FEEDBACK.txt').exists():print((O/'FEEDBACK.txt').read_text())
dump('HEARTBEAT.json',dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase='generate private pinned registration',accepted_added=0))
base=(C/'integration700_bf16_conv_registration1.py').read_text(); constants={}
for node in ast.parse(base).body:
 if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name):
  try:constants[node.targets[0].id]=ast.literal_eval(node.value)
  except (ValueError,TypeError):pass
review=json.loads((N/'PARENT_REVIEW.json').read_text());receipt=pathlib.Path(review['checker']);assert sha(receipt)==review['checker_sha256'];k=tuple(review['key'])
pins=json.loads((N/'FROZEN.json').read_text())['sha256'];excluded={}
for p in list(pins):
 if p.endswith('/ops_transformer/__init__.py'):excluded[p]=dict(historical_sha256=pins.pop(p),reason='Publication routing only; historical FROZEN unchanged')
# Runtime closure previously pinned by published registration; preserve hashes, add only
# files actually loaded in the approved masked execution plus startup hooks.
old=json.loads((C/'FROZEN_PINS.json').read_text());loaded={}
for p in json.loads((N/'result/EXECUTION.json').read_text())['loaded_files']:
 if p.startswith('/dev/'):continue
 p=str(pathlib.Path(p).resolve());h=old.get(p,pins.get(p));assert h is not None,p
 if p in pins:assert pins[p]==h,p
 pins[p]=h;loaded[p]=h
for p,h in old.items():
 if p.endswith('.pth') or p.endswith('/pyvenv.cfg') or p.endswith('/fla_npu_opp_env.py'):pins[p]=h
for p in [N/'FROZEN.json',N/'PARENT_REVIEW.json',N/'GUARD_RECEIPT.json',N/'NATIVE_PROOF_SUMMARY.json']+list((N/'result').glob('*.json')):
 pins[str(p)]=sha(p)
for name,v in json.loads((N/'result/CAPTURE.json').read_text()).items():
 if v.get('stored'):
  p=N/'result'/(name+'.bin');assert sha(p)==v['sha256'];pins[str(p)]=v['sha256']
local=[N/'resource_gate.py',B/'ascend_baselines.py',B/'ascend_math_more.py',R/'bf16_local_gemm18/binding.py',N/'adapter.py',B/'api.py',R/'bf16_moe_hadamard_parent10/adapter.py']
for p in local:assert str(p) in pins,p
# Manifest ADT implementation is CPU-only and participates in raw-call validation.
adt=W/'TileOPs/src/tileops/manifest/values.py';pins[str(adt)]=sha(adt)
dump('FROZEN_PINS.json',pins);dump('PIN_EXCLUSIONS.json',excluded)
params=review['params'];raw={n:params[n] for n in ('layout','activation','out_dtype')}
indices=dict(K=7168,E=32,N=4096,D='bfloat16',R=4096,**raw)
constants.update(MANIFEST=str(O/'FROZEN_PINS.json'),MANIFEST_HASH=sha(O/'FROZEN_PINS.json'),LOCAL_FILES=list(map(str,local)),CASE_PARAMS={k:json.dumps(params,sort_keys=True,separators=(',',':'))},CASE_INDICES=indices,ENVELOPES={k[2]:json.loads((N/'CASES.json').read_text())[0]['envelope']},LOADED_VENDOR=loaded)
# Runtime checker has already imported Torch before constructing candidates. Match
# the approved post-import environment, including original FLA vendor entries.
constants['ENV'].update(json.loads((N/'result/RUNTIME.json').read_text())['environment'])
head='"""Exact CPU-lazy masked MoE registration; original reviewed native route."""\nimport builtins, hashlib, json, os, types, tempfile, shutil, sys\nfrom pathlib import Path\nfrom types import MappingProxyType\n'
for name in ('ROOT','MANIFEST','MANIFEST_HASH','LOCAL_FILES','CASE_PARAMS','CASE_INDICES','ENVELOPES','ENV','LOADED_VENDOR','REQUIRED_LOADED'):head+=name+' = '+pprint.pformat(constants[name],width=120)+'\n'
body=base[base.index('ROOT = Path(ROOT)'):]
start=body.index('def _canonical(');end=body.index('def _in_guard_group(')
body=body[:start]+(O/'case_and_load.txt').read_text()+'\n\n'+body[end:]
body=body.replace("    from benchmarks.ascend_baselines import Candidate, UnsupportedCase\n", "    private_cpu = _load(_snapshot())\n    Candidate = private_cpu['benchmarks.ascend_baselines'].Candidate\n    UnsupportedCase = private_cpu['benchmarks.ascend_baselines'].UnsupportedCase\n")
start=body.index('        def owned_factory(');end=body.index('            _loaded_vendor_pins(require=True)',start)
body=body[:start]+'''        try:
            child = _native_candidate(private, frozen, owned)
'''+body[end:]
body=body.replace("    impl = types.SimpleNamespace(run=run,args=None,reset=None,noncomparable_reason=None)","    impl = private_cpu['benchmarks.api'].Implementation(run)")
body=body.replace('BF16_CONV','BF16_MOE_MASKED').replace('BF16 Conv1','BF16 masked MoE').replace('bf16-conv-registration1-','bf16-moe-masked-registration1-').replace('one canonical BF16 Conv1 case','one canonical BF16 masked MoE case')
body=body.replace("    if sys.prefix != '/root/dyq/workspace/.venv':", "    if sys.executable != '/root/dyq/workspace/.venv/bin/python' or sys.prefix != '/root/dyq/workspace/.venv':")
(O/'integration700_bf16_moe_masked_registration1.py').write_text(head+body)
recipe=(C/'run_environment.sh').read_text().replace('BF16_CONV','BF16_MOE_MASKED').replace('python3 -S -', '/usr/bin/python3 -I -S -')
# Original shell recipe sets vendor-only OPP. Torch expands it as captured; do not
# source/execute the recipe in this CPU worker.
recipe=recipe.replace('exec "$@"', '''if [ "${1:-}" != /root/dyq/workspace/.venv/bin/python ]; then
  echo 'Original /root/dyq/workspace/.venv/bin/python is required' >&2; exit 2
fi
exec "$@"''')
(O/'run_environment.sh').write_text(recipe)
dump('CONTRACTS.json',dict(accepted_added=0,key=k,report_params=params,raw_call_indices=indices,raw_call_params=raw,raw_layout_adt='MGroupedLayout',tensor_names=['a','b','layout_metadata','output'],envelope=constants['ENVELOPES'][k[2]],pins=len(pins),publication_exclusion=excluded,seed=review['seed'],numerical_evidence=review,algorithm='Unmodified recovery bridge, parent10 V5Binding.moe and original binding.dense_fp32. ENK b; per-expert counts; 3D a/output; BF16 lossless FP32 widening; original output dtype.',private_loader='Pinned bytes with private import hooks; private importlib util routes original dynamic binding load; private parent loader; no sys.modules substitution'))
print('generated',len(pins),'pins')
