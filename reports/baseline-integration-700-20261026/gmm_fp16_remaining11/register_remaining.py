from pathlib import Path
import fcntl,json,hashlib,datetime,sys,ast
W=Path('/opt/tileops-all-pairs/workspace');P=W/'reports/baseline-integration-700-20261026/gmm_fp16_remaining11';name=sys.argv[1];Q=P/name
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads((Q/'representative.json').read_text());assert r['status']=='verified' and r['inputs_preserved'] and r['verification']['checked_outputs']==r['verification']['total_outputs']
assert (Q/'guard.exit').read_text().strip()=='0'
root=W/'TileOPs/benchmarks/ascend_all_pairs/ops_transformer';module=root/'integration700_gmm_fp16_remaining11.py';op,cid=r['op'],r['case_id'];tag='_gmm_fp16_remaining11_'+name
function=f'''
def verified_{name}(case):
    assert case.id == {cid!r}
    path=Path({str(Q/'remaining_binding.py')!r})
    spec=importlib.util.spec_from_file_location({tag!r},path)
    m=importlib.util.module_from_spec(spec)
    sys.path.insert(0,str(path.parent))
    try:spec.loader.exec_module(m)
    finally:sys.path.remove(str(path.parent))
    return m.remaining(case)
'''
append=f'''
# {tag}: complete canonical verification, exact key only.
from .integration700_gmm_fp16_remaining11 import verified_{name} as {tag}_factory
{tag}_old = PROVIDERS.get({op!r})
def {tag}_dispatch(case):
    if case.id == {cid!r}: return {tag}_factory(case)
    return {tag}_old(case)
PROVIDERS[{op!r}] = {tag}_dispatch
'''
# CPU-only execution of the dispatch AST with sentinel factories establishes both branches.
from types import SimpleNamespace
node=next(n for n in ast.parse(append).body if isinstance(n,ast.FunctionDef));ns={tag+'_factory':lambda c:'tested',tag+'_old':lambda c:'preserved'}
exec(compile(ast.Module(body=[node],type_ignores=[]),'<dispatch-proof>','exec'),ns)
assert ns[tag+'_dispatch'](SimpleNamespace(id=cid))=='tested';assert ns[tag+'_dispatch'](SimpleNamespace(id='unrelated-key'))=='preserved'
with open('/tmp/baseline700-provider-ops_transformer.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);init=root/'__init__.py';before=init.read_bytes();assert tag not in before.decode()
 (Q/'registry_before.py').write_bytes(before);(Q/'module_before.py').write_bytes(module.read_bytes())
 new_module=module.read_text()+function;new_init=before.decode()+append
 compile(new_module,str(module),'exec');compile(new_init,str(init),'exec')
 module.write_text(new_module);init.write_text(new_init)
 proof={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'key':['ops-transformer',op,cid],'factory':'integration700_gmm_fp16_remaining11.verified_'+name,'tested_binding':str(Q/'remaining_binding.py'),'tested_binding_sha256':sha(Q/'remaining_binding.py'),'v5_binding_sha256':sha(Q/'v5_fp32_bias.py'),'checker_sha256':sha(Q/'representative.json'),'guard_exit':0,'checked_outputs':r['verification'],'inputs_preserved':r['inputs_preserved'],'module_sha256':sha(module),'registry_before_sha256':hashlib.sha256(before).hexdigest(),'registry_after_sha256':sha(init),'prior_registry_prefix_preserved':init.read_bytes().startswith(before),'dispatch_AST_exact_key_test':True,'dispatch_AST_preserves_other_keys':True}
 (Q/'REGISTRATION_PROOF.json').write_text(json.dumps(proof,indent=2));(Q/'registered_function.py').write_text(function);(Q/'registration_append.py').write_text(append)
print(json.dumps(proof))
