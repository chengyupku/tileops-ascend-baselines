"""Audit Exp under local runtime core limit without replacing baseline acceptance."""
import json,hashlib,datetime
from pathlib import Path
import xml.etree.ElementTree as ET
r=Path(__file__).resolve().parent
for name in ['exp-numerical','exp-boundaries','exp-manifest-v1','exp-public']:
 g=json.loads((r/'requests'/name/'guard.json').read_text());assert g['exit_code']==0 and g['performance_valid'],name
rows=[json.loads(l) for l in (r/'exp-manifest-v1.jsonl').read_text().splitlines()]
pairs=[v for v in json.loads((r/'baseline-scope.json').read_text())['pairs'] if v['op']=='ExpFwdOp']
assert len(rows)==5 and {v['case_id'] for v in rows}=={v['case'] for v in pairs}
for v in rows:
 assert v['correct'] and v['input_unchanged'] and v['verification']['checked_outputs'] and len(v['latency_samples_ms'])==15 and len(v['eager']['latency_samples_ms'])==15
 assert v['runtime_limit']=={'cube_core_num':32,'vector_core_num':64}
 for p,h in v['source_manifest'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
for name,n in [('exp-numerical',3),('exp-boundaries',33)]:
 d=json.loads((r/(name+'.json')).read_text());assert len(d['rows'])==n and all(v['correct'] and v['input_unchanged'] for v in d['rows'])
 for p,h in d['sources'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
 if name=='exp-numerical':assert all(v['changed_input_checked'] and v['negative_control_rejected'] for v in d['rows'])
suites=list(ET.parse(r/'exp-public.xml').getroot().iter('testsuite'));assert sum(int(s.get('tests',0)) for s in suites)==4;assert all(int(s.get('errors',0))+int(s.get('failures',0))==0 for s in suites)
ratios=[v['eager']['latency_ms']/v['latency_ms'] for v in rows];assert min(ratios)>=.8
report={'op':'ExpFwdOp','completed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'manifest_correct':5,'same_host_eager_80_percent_passes':5,'minimum_eager_ratio':min(ratios),'boundary_checks':33,'public_tests':4,'numerical_checks':'FP16/BF16 all 65536 patterns; 73743 FP32 values; original contract, input hashes, changed inputs and negative control','runtime_limit':{'cube':32,'vector':64},'registered_baseline_pairs':len(pairs),'full_registered_baseline_acceptance':False,'source_manifest':rows[0]['source_manifest']}
(r/'exp-REPORT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
p=r.parent/'elementwise_serial_20261030/STATE.json';s=json.loads(p.read_text())
for v in s['operators']:
 if v['op']=='ExpFwdOp':v.update(local32_validated=True,status='baseline_pending',detail=f"本地32AIC：5/5正确且同机eager达80%，最差{min(ratios):.3f}；33形状+4公开+全位模式通过，登记baseline待补",evidence=str(r/'exp-REPORT.json'))
s['updated_at']=report['completed_at'];t=p.with_suffix('.tmp');t.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n');t.replace(p)
print(json.dumps(report,ensure_ascii=False,indent=2))
