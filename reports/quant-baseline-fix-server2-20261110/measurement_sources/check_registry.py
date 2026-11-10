"""CPU-only exact registry/source checks; no measurement reruns."""
import pathlib,json,sys
P=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(P/'delivery'))
from quant_baselines.registry import available,verify_reference_sources
rows=json.loads((P/'delivery/verified_cases.json').read_text())
for row in rows:
 assert row['variant'] in available(row['op'],row['case'],row['parameters'])
 altered=dict(row['parameters']);altered['_different_contract']=True
 try:available(row['op'],row['case'],altered)
 except ValueError:pass
 else:raise AssertionError('Changed parameters accepted')
assert available('UNKNOWN','UNKNOWN',{})==[]
assert verify_reference_sources(P/'delivery/reference_snapshot')
assert not any(n in sys.modules for n in ('torch','torch_npu'))
r={'registered_records_checked':len(rows),'changed_parameters_rejected':True,'unknown_case_excluded':True,'original_sources_match':True,'no_torch_or_npu_imports':True}
(P/'REGISTRY_CPU_CHECKS.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
