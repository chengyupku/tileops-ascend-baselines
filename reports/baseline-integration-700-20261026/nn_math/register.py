"""Publish only complete checker successes under shared provider lock."""
import ast,fcntl,hashlib,json,sys,difflib
from pathlib import Path
P=Path(__file__).resolve().parent;W=P.parents[2];attempt=sys.argv[1];base=sys.argv[2];module=base.replace('-','_');out=P/attempt
r=json.loads((out/'checker.json').read_text());assert r['status']=='verified';v=r['verification'];assert v['checked_outputs']==v['total_outputs'] and v['checked_outputs']>0
native=(out/'native.jsonl').read_text()+((out/'OWNERSHIP.json').read_text() if (out/'OWNERSHIP.json').exists() else '');assert any(v in native for v in ('allpairs_nn_impl','classicpilot_nn_impl','allpairs_math_math_impl')) and 'kernel_name=te_' in native, 'Require own graph compilation and launch receipt'
source=out/'adapter.py';digest=hashlib.sha256(source.read_bytes()).hexdigest();assert digest in r['source_sha256'].values()
provider=W/'TileOPs/benchmarks/ascend_all_pairs'/module;name='integration700_nn_math_'+attempt;dest=provider/(name+'.py');dest.write_bytes(source.read_bytes());key=(r['op'],r['case_id'])
with open('/tmp/baseline700-provider-'+module+'.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);init=provider/'__init__.py';before=init.read_text();backup=out/'provider_init.before.py';backup.write_text(before)
 addition=f'''\n# Exact numerical receipt: baseline-integration-700-20261026/nn_math/{attempt}\nfrom .{name} import make as _nn_math_{attempt}\n_nn_math_old_{attempt} = PROVIDERS.get({key[0]!r})\ndef _nn_math_route_{attempt}(case, _prior=_nn_math_old_{attempt}):\n    if case._call.signature.name == {key[0]!r} and case.id == {key[1]!r}:\n        return _nn_math_{attempt}(case)\n    if _prior is not None:\n        return _prior(case)\n    from benchmarks.ascend_baselines import UnsupportedCase\n    raise UnsupportedCase('No accepted exact key')\nPROVIDERS[{key[0]!r}] = _nn_math_route_{attempt}\n'''
 assert f'from .{name} import' not in before
 after=before+addition;ast.parse(after);init.write_text(after)
 (out/'registration.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=str(init),tofile=str(init))))
(out/'REGISTRATION.json').write_text(json.dumps(dict(key=[base,*key],factory=f'benchmarks.ascend_all_pairs.{module}.{name}.make',module=str(dest),sha256=digest,checker=str(out/'checker.json'),backup=str(backup),patch=str(out/'registration.patch'),device_free_check='AST parsed; exact-key closure and existing PROVIDERS preserved',environment='ASCEND_CLASSIC_NN_VENDOR and ASCEND_CUSTOM_OPP_PATH = /root/dyq/workspace/.cache/all-pairs/ops_nn/install3/vendors/allpairs_nn'),indent=2))
print(dest)
