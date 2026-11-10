"""CPU-only adaptation of the published masked registration; no device imports."""
import ast, builtins, datetime, hashlib, json, pprint, sys
from pathlib import Path
sys.dont_write_bytecode=True
W=Path('/root/dyq/workspace'); O=Path(__file__).resolve().parent; B=O.parent; E=B/'bf16_moe_down_full3'
T=W/'TileOPs/benchmarks/ascend_all_pairs/ops_transformer/integration700_bf16_moe_masked_registration1.py'
original=builtins.__import__
def protected(name,*a,**kw):
    if name.split('.')[0] in ('torch','torch_npu','acl'):raise AssertionError(name)
    return original(name,*a,**kw)
builtins.__import__=protected
sys.path[:0]=[str(W/'TileOPs/src'),str(W/'.venv/lib/python3.12/site-packages')]
from tileops.manifest import load_manifest,load_adts,load_workloads
from tileops.manifest.plan import entry_plan
from tileops.manifest.workload import instantiate

def plain(v):
    if getattr(v,'adt',None)=='MGroupedLayout':return {v.kind:plain(v.fields)}
    if isinstance(v,dict):return {k:plain(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [plain(x) for x in v]
    return v

def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def heartbeat(phase):
    (O/'HEARTBEAT.json').write_text(json.dumps(dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase=phase,npu_runs=0)))
rows=json.loads((E/'CASES.json').read_text()); keys={tuple(r['key']) for r in rows}
op='MoEGroupedGemmFwdOp'; plan=entry_plan(op,load_manifest()[op],load_adts(),resolve=False)
calls={c.case_id:c for r in load_workloads(op) for d in r.get('dtype_cases') or [{}] if (c:=instantiate(plan,r,d)).case_id in {k[2] for k in keys}}
assert len(calls)==3
params={tuple(r['key']):json.dumps(r['manifest_params'],sort_keys=True,separators=(',',':')) for r in rows}
indices={k:plain(calls[k[2]].indices) for k in sorted(keys)}
envelopes={r['key'][2]:r['envelope'] for r in rows}
source=T.read_text(); tree=ast.parse(source)
values={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.lineno<200 and n.targets[0].id in ('LOCAL_FILES','ENV','REQUIRED_LOADED')}
local=[x.replace('bf16_moe_masked_recovery1','bf16_moe_down_full3') for x in values['LOCAL_FILES']]
pins=json.loads((E/'FROZEN.json').read_text())['sha256']
# Publication only: exclude exact registry initializer/router files, never math.
excluded={p:d for p,d in pins.items() if p in {str(W/'TileOPs/benchmarks/ascend_all_pairs'/q) for q in ('__init__.py','registry.py','ops_transformer/__init__.py','ops_nn/__init__.py','ops_math/__init__.py')}}
for p in excluded:del pins[p]
loaded={}; origins={}
template_pins=json.loads((B/'bf16_moe_masked_registration1/FROZEN_PINS.json').read_text())
for r in rows:
    ex=json.loads((E/r['key'][2]/'result/EXECUTION.json').read_text())
    for p in ex['loaded_files']:
        if p.startswith('/dev/'):continue
        p=str(Path(p).resolve());digest=sha(p)
        if p in pins:assert pins[p]==digest,(p,'frozen mismatch')
        assert template_pins[p]==digest,(p,'reviewed runtime changed')
        if p in loaded:assert loaded[p]==digest
        loaded[p]=digest;origins[p]='successful down EXECUTION.loaded_files; unchanged published-template runtime digest and current CPU hash verified'
    for p,d in ex['mapped_host_hashes'].items():assert loaded[str(Path(p).resolve())]==d
pins.update(loaded)
# Extend original FROZEN with reviewed passive bootstrap/ADT pins from the
# published template. These are fixed historical digests, not runtime repins.
template_pins=json.loads((B/'bf16_moe_masked_registration1/FROZEN_PINS.json').read_text())
bootstrap={p:d for p,d in template_pins.items() if p.endswith('.pth') or p.endswith('/fla_npu_opp_env.py') or p.endswith('/tileops/manifest/values.py')}
for p,d in bootstrap.items():
    assert sha(p)==d,(p,'reviewed bootstrap changed')
    if p in pins:assert pins[p]==d
    pins[p]=d
# Pin successful per-case receipts and original full outputs using their
# previously recorded artifact digests; exclude mutable orchestration files.
artifacts=json.loads((E/'FILEHASHES.json').read_text())
for rel,d in artifacts.items():
    if any(rel.startswith(r['key'][2]+'/result/') for r in rows):
        p=str(E/rel)
        if p in pins:assert pins[p]==d
        pins[p]=d

for p in local:assert p in pins,p
for p in (E/'PARENT_REVIEW.json',E/'FROZEN.json',E/'CASES.json'):
    pins[str(p)]=sha(p)
(O/'FROZEN_PINS.json').write_text(json.dumps(pins,indent=2,sort_keys=True)+'\n')
(O/'PIN_PROVENANCE.json').write_text(json.dumps(dict(original_frozen=str(E/'FROZEN.json'),original_frozen_hash=sha(E/'FROZEN.json'),excluded_publication_routes=excluded,loaded_file_origins=origins,loaded_historical_hash_crosscheck_count=len(loaded),reviewed_bootstrap_pins=bootstrap,template=str(T),template_sha256=sha(T)),indent=2)+'\n')
replace={'MANIFEST':str(O/'FROZEN_PINS.json'),'MANIFEST_HASH':sha(O/'FROZEN_PINS.json'),'LOCAL_FILES':local,'CASE_PARAMS':params,'CASE_INDICES':indices,'ENVELOPES':envelopes,'LOADED_VENDOR':loaded}
lines=source.splitlines(keepends=True)
for n in reversed(tree.body):
    if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in replace and n.lineno<200:
        name=n.targets[0].id;lines[n.lineno-1:n.end_lineno]=[name+' = '+pprint.pformat(replace[name],width=110)+'\n']
source=''.join(lines)
source=source.replace('BF16_MOE_MASKED_GUARD_OWNER','BF16_GEMM_GUARD_OWNER').replace('BF16_MOE_MASKED_RUN_ROOT','BF16_GEMM_RUN_ROOT')
source=source.replace("getattr(value,'kind',None)=='masked'","getattr(value,'kind',None)=='contiguous'")
source=source.replace("getattr(layout,'kind',None)!='masked'","getattr(layout,'kind',None)!='contiguous'")
source=source.replace("if _canonical(layout.fields)!='{\"max_m\":128}': return False","if _canonical(layout.fields)!=_canonical(p['layout']['contiguous']): return False")
source=source.replace('_canonical(CASE_INDICES)','_canonical(CASE_INDICES[_key(case,baseline)])')
source=source.replace("layout=FrozenADT('MGroupedLayout','masked',_deepfreeze({'max_m':128}))","layout=FrozenADT('MGroupedLayout','contiguous',_deepfreeze(p['layout']['contiguous']))")
source=source.replace('raw=dict(layout=layout,activation=None,out_dtype=None)',"raw=dict(layout=layout,activation=p['activation'],out_dtype=p['out_dtype'])")
source=source.replace('indices=dict(CASE_INDICES,layout=layout)','indices=dict(CASE_INDICES[_key(case)],layout=layout)')
source=source.replace('bf16-moe-masked-registration1-','bf16-moe-down-registration3-').replace('masked MoE','down MoE').replace('exact one canonical BF16','exact three canonical BF16')
source=source.replace('Exact CPU-lazy masked','Exact CPU-lazy down')
# Original Case may be mutated after construction; the captured contract remains
# immutable, while a changed caller is rejected instead of changing mathematics.
(O/'integration700_bf16_moe_down_registration3.py').write_text(source)
(O/'CONTRACTS.json').write_text(json.dumps([dict(r,call_indices=indices[tuple(r['key'])]) for r in rows],indent=2)+'\n')
heartbeat('module_and_pins_prepared')
with (O/'FEEDBACK.txt').open('a') as f:f.write('\nCPU preparation: three real manifest indices extracted; exact down contracts/envelopes and successful native paths adapted from published template. No NPU, no shared writes.\n')
print(json.dumps(dict(cases=len(keys),pins=len(pins),loaded_files=len(loaded),excluded_routes=len(excluded))))
