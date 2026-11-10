"""Audit independent correctness and same-host limited-core performance evidence."""
import json,hashlib,datetime
from pathlib import Path
import xml.etree.ElementTree as ET
r=Path(__file__).resolve().parent
for label in ['abs-probe-v4','abs-boundaries','abs-manifest-v1','abs-public']:
 g=json.loads((r/'requests'/label/'guard.json').read_text());assert g['exit_code']==0 and g['performance_valid'],label
rows=[json.loads(l) for l in (r/'abs-manifest-v1.jsonl').read_text().splitlines()]
pairs=[v for v in json.loads((r/'baseline-scope.json').read_text())['pairs'] if v['op']=='AbsFwdOp']
assert len(rows)==5 and {v['case_id'] for v in rows}=={v['case'] for v in pairs}
for v in rows:
 assert v['correct'] and v['input_unchanged'] and v['verification']['checked_outputs']
 assert len(v['latency_samples_ms'])==len(v['eager']['latency_samples_ms'])==15
 assert v['runtime_limit']=={'cube_core_num':32,'vector_core_num':64}
 for p,h in v['source_manifest'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
probe=json.loads((r/'abs-probe-v4.json').read_text());assert len(probe['checks'])==24 and all(v['passed'] for v in probe['checks'])
module=r.parents[3]/'TileOPs/src/tileops/kernels/ascend/elementwise/abs.py'
assert hashlib.sha256(module.read_bytes()).hexdigest()==probe['module_sha256']
b=json.loads((r/'abs-boundaries.json').read_text());assert len(b['rows'])==36 and all(v['correct'] for v in b['rows'])
assert all(v['input_unchanged'] for v in b['rows'] if 'shape' in v)
for p,h in b['sources'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
suites=list(ET.parse(r/'abs-public.xml').getroot().iter('testsuite'));n_public=sum(int(s.get('tests',0)) for s in suites)
assert n_public==8 and all(int(s.get('errors',0))+int(s.get('failures',0))+int(s.get('skipped',0))==0 for s in suites)
ratios=[v['eager']['latency_ms']/v['latency_ms'] for v in rows];assert min(ratios)>=.8
report={'op':'AbsFwdOp','completed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'manifest_correct':5,'same_host_eager_80_percent_passes':5,'minimum_eager_ratio':min(ratios),'boundary_checks':36,'public_tests':n_public,'probe_checks':24,'numerical_checks':'FP16/BF16 all 65536 patterns, FP32 special values, all integer dtypes including signed minima','runtime_limit':{'cube':32,'vector':64},'registered_baseline_pairs':len(pairs),'full_registered_baseline_acceptance':False,'source_manifest':rows[0]['source_manifest']}
(r/'abs-REPORT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
p=r.parent/'elementwise_serial_20261030/STATE.json';s=json.loads(p.read_text())
for v in s['operators']:
 if v['op']=='AbsFwdOp':v.update(local32_validated=True,status='baseline_pending',detail=f"本地32AIC：5/5正确且同机eager达80%，最差{min(ratios):.3f}；36边界+8公开+24探针通过，登记baseline待补",evidence=str(r/'abs-REPORT.json'))
s['updated_at']=report['completed_at'];t=p.with_suffix('.tmp');t.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n');t.replace(p)
print(json.dumps(report,ensure_ascii=False,indent=2))
