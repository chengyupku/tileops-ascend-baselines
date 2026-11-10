from pathlib import Path
from types import SimpleNamespace,ModuleType
import json,hashlib,sys,runpy,importlib.util
from datetime import datetime
T=Path(__file__).resolve().parent
class UnsupportedCase(Exception):pass
m=ModuleType('benchmarks.ascend_baselines');m.UnsupportedCase=UnsupportedCase;sys.modules['benchmarks']=ModuleType('benchmarks');sys.modules['benchmarks.ascend_baselines']=m
ns=runpy.run_path(str(T/'exact_registration.py'));make=ns['make_candidate'];rows=json.loads((T/'CASES.json').read_text())
def case(r):return SimpleNamespace(id=r['key'][2],_call=SimpleNamespace(signature=SimpleNamespace(name=r['key'][1]),indices=dict(r['manifest_params'])))
bmm=next(r for r in rows if r['key'][1]=='BmmFwdOp');valid=case(bmm)
# Fake only the numerical constructor; exercise exact key/params/evidence/hash gates.
original_spec=importlib.util.spec_from_file_location;original_module=importlib.util.module_from_spec
seen=[]
importlib.util.spec_from_file_location=lambda *a:SimpleNamespace(loader=SimpleNamespace(exec_module=lambda m:None))
importlib.util.module_from_spec=lambda s:SimpleNamespace(make_candidate=lambda c:seen.append(c) or 'constructor_stub')
assert make(valid)=='constructor_stub' and seen==[valid]
rejected=[]
for r in rows:
 if r is bmm:continue
 try:make(case(r))
 except UnsupportedCase:rejected.append(r['key'][2])
 else:raise AssertionError('unverified key accepted')
invalid=case(bmm);invalid._call.indices['B']=1
try:make(invalid)
except UnsupportedCase:rejected.append('wrong_batch')
else:raise AssertionError('wrong batch accepted')
assert len(seen)==1 and 'torch' not in sys.modules
importlib.util.spec_from_file_location=original_spec;importlib.util.module_from_spec=original_module
result=json.loads((T/'RESULT.json').read_text());assert all(not x for x in result['pins_changed'].values());assert result['accepted_added']==0
summary=[]
for r in result['rows']:
 C=T/r['key'][2];m=json.loads((C/'GUARD_FINAL.json').read_text());stamp=lambda k:datetime.fromisoformat(m[k]).timestamp()
 timing=dict(queue_to_lock_s=stamp('acquired_at')-stamp('requested_at'),lock_to_finish_s=stamp('updated_at')-stamp('acquired_at'))
 r['guard_timing']=timing
 summary.append(dict(key=r['key'],status=r['status'],guard_exit=r['guard_exit_code'],native_entries=r['native_entries'],timings=r['timings'],guard_timing=timing))
(T/'FINAL_CHECKS.json').write_text(json.dumps(dict(pins_unchanged_per_case=True,exact_factory_constructor_stub_pass=True,unverified_and_wrong_params_rejected=rejected,torch_imported=False,numerical_replays=0,rows=summary,startup_events=['dgrad attempt1 parameter-observer assertion before input generation; fixed and retained','Bmm publication-boundary interruption in torch import before input generation; retained'],scope_writes='task directory only; no shared registry/memory changes'),indent=2))
(T/'RESULT.json').write_text(json.dumps(result,indent=2))
delivery=json.loads((T/'DELIVERY.json').read_text());delivery['startup_events_preserved']=['moe-dgrad-startup-attempt1','square-2k-publication-startup'];delivery['summary']=summary;delivery['checks'].append('CPU exact-key factory gates with stubbed numerical constructor; rejects unverified cases and wrong batch');(T/'DELIVERY.json').write_text(json.dumps(delivery,indent=2))
(T/'FILEHASHES.json').write_text(json.dumps({str(p.relative_to(T)):hashlib.sha256(p.read_bytes()).hexdigest() for p in T.rglob('*') if p.is_file() and p.name!='FILEHASHES.json'},indent=2))
print(json.dumps(summary))
