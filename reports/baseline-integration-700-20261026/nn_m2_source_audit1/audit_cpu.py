"""Read-only CPU evidence audit; writes only this private packet. No torch imports."""
import ast, difflib, hashlib, json
from pathlib import Path
W=Path('/root/dyq/workspace'); B=W/'reports/baseline-integration-700-20261026'; P=Path(__file__).resolve().parent
R=B/'nn_recovery02/matrix_05_01'; C=json.loads((R/'checker.json').read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,obj): (P/name).write_text(json.dumps(obj,indent=2)+'\n')
def record(p): return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
before=B/'nn_parent_last3/provider_init.before.py'; after=B/'nn_registration02/provider_before_e4fa1c09f3215487fcd288341915395e50fbd6f7428d79b3f2b001b17b36b31c.py'; added=W/'TileOPs/benchmarks/ascend_all_pairs/ops_nn/integration700_nn_parent_last3.py'
key='TileOPs/benchmarks/ascend_all_pairs/ops_nn/__init__.py'; ak=str(added.relative_to(W))
assert sha(before)==C['provider_source_sha256'][key]
assert sha(after)==C['provider_source_sha256_after'][key]
assert sha(added)==C['provider_source_sha256_after'][ak]
assert after.read_bytes().startswith(before.read_bytes())
suffix=after.read_text()[len(before.read_text()):]
(P/'PROVIDER_RACE.diff').write_text(''.join(difflib.unified_diff(before.read_text().splitlines(True),after.read_text().splitlines(True),fromfile=str(before),tofile=str(after))))
changes=[k for k in C['provider_source_sha256'].keys()|C['provider_source_sha256_after'].keys() if C['provider_source_sha256'].get(k)!=C['provider_source_sha256_after'].get(k)]
assert set(changes)=={key,ak}
# Execute ONLY the appended registry statements with explicit import stub.
import types,builtins
fake=types.SimpleNamespace(_KEYS={('GemmW4A16FwdOp','decode-long-k-float16')},make=lambda c:'w4-new')
def stub_import(name,globals=None,locals=None,fromlist=(),level=0):
 assert level==1 and fromlist==('integration700_nn_parent_last3',)
 return types.SimpleNamespace(integration700_nn_parent_last3=fake)
sentinel=lambda c:'gemm-original';old=lambda c:'w4-original'
ns={'__builtins__':{**vars(builtins),'__import__':stub_import},'PROVIDERS':{'GemmFwdOp':sentinel,'GemmW4A16FwdOp':old}}
exec(compile(suffix,'race_suffix','exec'),ns)
assert ns['PROVIDERS']['GemmFwdOp'] is sentinel
assert ns['PROVIDERS']['GemmW4A16FwdOp'](types.SimpleNamespace(id='unmatched'))=='w4-original'
# All recorded actual sources, including six native shared objects, are read as bytes only.
checks={}
for field in ('source_sha256','oracle_source_sha256'):
 bad=[]
 for rel,expected in C[field].items():
  p=W/rel
  if not p.is_file() or sha(p)!=expected: bad.append(rel)
 checks[field]={'count':len(C[field]),'mismatches_or_missing':bad}
 assert not bad
assert sha(W/'TileOPs/scripts/ascend_pair_check.py')==C['verifier_sha256']
# Check installed principal kernel sources byte-for-byte against original baseline.
pairs=[]
for leaf in ('mat_mul_v3.cpp','arch35/mat_mul_v3.cpp'):
 a=W/'.cache/ascend-classic/native-pilot-20261020/ops-nn/install/vendors/classicpilot_nn/op_impl/ai_core/tbe/classicpilot_nn_impl/ascendc/mat_mul_v3'/leaf
 b=W/'ascend_baselines/ops-nn/matmul/mat_mul_v3/op_kernel'/leaf
 assert a.read_bytes()==b.read_bytes();pairs.append({'installed':str(a),'original':str(b),'sha256':sha(a)})
events=[json.loads(x) for x in (R/'native.jsonl').read_text().splitlines()]
logs={e['file']:Path(e['file']).read_text().splitlines() for e in events if Path(e['file']).exists()}
raw_matches=[]
for e in events:
 lines=logs.get(e['file'],[]);exact=e['line']<=len(lines) and lines[e['line']-1]==e['text']
 raw_matches.append(exact)
selected=[e for e in events if 'te_matmulv3' in e['text']]
write('NATIVE_EVIDENCE.json',{'events':selected,'all_45_captured_lines_match_raw':all(raw_matches),'matched':sum(raw_matches),'total':len(events),'raw_files':[record(Path(p)) for p in logs], 'kernel_elf_retained':False,'logged_kernel_sha256':'a24e62a45ed206deffde96c22f1dab7bd05ba5af2f313cb74ba707fdfddaf301','artifact_search_scope':str(R),'matching_artifacts':[str(p) for p in R.rglob('*e1f1ffb460145f29*')]})
files=[R/f for f in ('checker.json','adapter.py','case.log','native.jsonl','native.summary.json','DISPATCH.json','GUARD_METADATA.json','GUARD_RESULT.json','CHILD_RESULT.json','CASE_STARTED.json')]
files += [B/'nn_recovery02/matrix_batch01'/f for f in ('GUARD_RESULT.json','COMMAND.json','guard.log')]
files += [before,after,added,B/'nn_parent_last3/REGISTRATION.json',B/'nn_recovery02/run_case.py',B/'nn_recovery02/batch_process.py',B/'nn_recovery02/launch_batch.py',W/'TileOPs/scripts/ascend_pair_check.py',W/'tools/baseline_protocol.py']
files += [Path(p) for p in logs]
write('EVIDENCE_HASHES.json',[record(p) for p in files])
write('SOURCE_RACE_RECONCILIATION.json',{'race_reconciled':True,'original_checker_status':C['status'],'changed_paths':changes,'before':record(before),'after':record(after),'added_module':record(added),'registration_at':json.loads((B/'nn_parent_last3/REGISTRATION.json').read_text())['at'],'exact_diff':'PROVIDER_RACE.diff','effect':'Append only GemmW4A16FwdOp wrapper. Added module top level imports stdlib and already-loaded UnsupportedCase, defines paths/keyset/functions; adapter load is lazy. No GemmFwdOp mutation or device action. Stub execution preserves GemmFwdOp identity and W4 fallback. run_case.py explicitly binds this exact adapter before candidate_factory assertion.','dependency_checks':checks,'original_kernel_pairs':pairs,'native_shared_objects':[{'path':k,'sha256':v} for k,v in C['source_sha256'].items() if k.endswith(('.so','.o'))],'limitations':['Current source hashes match all 2170 historical source records; no continuous historical filesystem monitor exists.','JIT ELF bytes are absent under recorded attempt; logged ELF hash cannot be independently rehashed.']})
# Freeze exact tested bytes. Runtime check pins every recorded candidate source, not the unrelated provider package.
frozen=P/('adapter_'+sha(R/'adapter.py')+'.py');frozen.write_bytes((R/'adapter.py').read_bytes())
write('DEPENDENCY_HASHES.json',C['source_sha256'])
template=(B/'nn_registration02/integration700_nn_registration02.py').read_text()
tree=ast.parse(template); node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_KEYS' for t in n.targets))
lines=template.splitlines();lines[node.lineno-1:node.end_lineno]=["_KEYS = frozenset([('GemmFwdOp', 'ds-v3-m2-qkv-a-bfloat16')])"]
module='\n'.join(lines)+'\n';module=module.replace('integration700_nn_registration02','integration700_nn_m2_source_audit1').replace('nn_registration02','nn_m2_source_audit1')
module=module.replace('    return _load().make(case)', '''    manifest = Path(__file__).with_name('DEPENDENCY_HASHES.json')
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != _DEPENDENCIES_SHA256:
        raise RuntimeError('Dependency manifest changed')
    import json
    for relative, expected in json.loads(manifest.read_text()).items():
        source = Path('/root/dyq/workspace') / relative
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise RuntimeError('Reviewed dependency changed: ' + relative)
    return _load().make(case)''')
module += '\n_DEPENDENCIES_SHA256 = '+repr(sha(P/'DEPENDENCY_HASHES.json'))+'\n'
(P/'integration700_nn_m2_source_audit1.py').write_text(module)
# Append patch includes explicit previous factory fallback; never apply here.
provider=W/'TileOPs/benchmarks/ascend_all_pairs/ops_nn/__init__.py';oldtext=provider.read_text()
append='''
# Exact registration: baseline-integration-700-20261026/nn_m2_source_audit1
from importlib.util import spec_from_file_location as _nn_m2_spec, module_from_spec as _nn_m2_module
_nn_m2_s = _nn_m2_spec('integration700_nn_m2_source_audit1', '/root/dyq/workspace/reports/baseline-integration-700-20261026/nn_m2_source_audit1/integration700_nn_m2_source_audit1.py')
_nn_m2 = _nn_m2_module(_nn_m2_s)
_nn_m2_s.loader.exec_module(_nn_m2)
_nn_m2_prior = PROVIDERS.get('GemmFwdOp')
def _nn_m2_route(case, _prior=_nn_m2_prior):
    if (case._call.signature.name, case.id) in _nn_m2._KEYS:
        return _nn_m2.make(case)
    if _prior is not None:
        return _prior(case)
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase('No exact GemmFwdOp route')
PROVIDERS['GemmFwdOp'] = _nn_m2_route
'''
(P/'REGISTRATION.patch').write_text(''.join(difflib.unified_diff(oldtext.splitlines(True),(oldtext+append).splitlines(True),fromfile='a/'+str(provider.relative_to(W)),tofile='b/'+str(provider.relative_to(W)))))
write('PATCH_BASE.json',record(provider))
print(json.dumps({'race_reconciled':True,'source_checks':checks,'raw_line_matches':sum(raw_matches),'raw_events':len(events),'packet':str(P)}))
