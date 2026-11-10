"""Audit local32 log/root validation; baseline acceptance remains separate."""
import argparse,json,hashlib,datetime
from pathlib import Path
import xml.etree.ElementTree as ET
p=argparse.ArgumentParser();p.add_argument('kind',choices=['log','sqrt','rsqrt','log1p']);p.add_argument('--measurement',required=True);a=p.parse_args()
r=Path(__file__).resolve().parent;op={'log':'LogFwdOp','sqrt':'SqrtFwdOp','rsqrt':'RsqrtFwdOp','log1p':'Log1pFwdOp'}[a.kind]
for label in [a.measurement,a.kind+'-edges',(a.kind+'-public-cpu-oracle' if a.kind=='log' else a.kind+'-public')]:
 g=json.loads((r/'requests'/label/'guard.json').read_text());assert g['exit_code']==0 and g['performance_valid'],label
rows=[json.loads(l) for l in (r/(a.measurement+'.jsonl')).read_text().splitlines()];pairs=[v for v in json.loads((r/'baseline-scope.json').read_text())['pairs'] if v['op']==op]
assert len(rows)==5 and {v['case_id'] for v in rows}=={v['case'] for v in pairs}
for v in rows:
 assert v['correct'] and v['input_unchanged'] and v['verification']['checked_outputs'] and len(v['latency_samples_ms'])==15 and len(v['eager']['latency_samples_ms'])==15
 assert v['runtime_limit']=={'cube_core_num':32,'vector_core_num':64}
 for path,h in v['source_manifest'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
edges=json.loads((r/(a.kind+'-edges.json')).read_text());assert all(v['passed'] for v in edges['checks']);assert len(edges['checks'])==(33 if a.kind=='log' else 30)
root=r.parents[3]/'TileOPs/src/tileops/kernels/ascend/elementwise'
for path,key in [(root/(a.kind+'.py'),'module_sha256'),(root/'_logroots.py','helper_sha256'),(r/'logroots_edges.py','runner_sha256')]:assert hashlib.sha256(path.read_bytes()).hexdigest()==edges[key]
suites=list(ET.parse(r/(a.kind+'-public.xml')).getroot().iter('testsuite'));assert sum(int(v.get('tests',0)) for v in suites)==4;assert all(int(v.get('errors',0))+int(v.get('failures',0))==0 for v in suites)
ratios=[v['eager']['latency_ms']/v['latency_ms'] for v in rows];assert min(ratios)>=.8
report={'op':op,'completed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'manifest_correct':5,'same_host_eager_80_percent_passes':5,'minimum_eager_ratio':min(ratios),'edge_checks':len(edges['checks']),'public_tests':4,'runtime_limit':{'cube':32,'vector':64},'registered_baseline_pairs':len(pairs),'full_registered_baseline_acceptance':False,'source_manifest':rows[0]['source_manifest'],'measurement':a.measurement}
if a.kind=='log':
 diagnostic=json.loads((r/'log-subnormal-diagnostic.json').read_text());assert diagnostic['tileops_matches_cpu'] and diagnostic['npu_eager_subnormal_is_negative_infinity']
 report['public_reference_correction']={'original_result':'3 passed, 1 failed at 1e-38 due to NPU eager FTZ','corrected_result':'4 passed with CPU FP64 reference, same inputs and tolerance','diagnostic':'log-subnormal-diagnostic.json','original_xml':'log-public-original-reference.xml','test_source_sha256':hashlib.sha256((r.parents[3]/'TileOPs/tests/ops/test_unary_math.py').read_bytes()).hexdigest()}
(r/(a.kind+'-REPORT.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
p=r.parent/'elementwise_serial_20261030/STATE.json';s=json.loads(p.read_text())
for v in s['operators']:
 if v['op']==op:v.update(local32_validated=True,status='baseline_pending',detail=f"本地32AIC：5/5正确且同机eager达80%，最差{min(ratios):.3f}；{len(edges['checks'])}边界+4公开通过，登记baseline待补",evidence=str(r/(a.kind+'-REPORT.json')))
s['updated_at']=report['completed_at'];t=p.with_suffix('.tmp');t.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n');t.replace(p)
print(json.dumps(report,ensure_ascii=False,indent=2))
