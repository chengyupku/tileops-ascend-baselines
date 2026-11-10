"""Audit a completed logical op against manifest, source hashes and guard receipts."""
import argparse,datetime,hashlib,json
from pathlib import Path
import xml.etree.ElementTree as ET
p=argparse.ArgumentParser();p.add_argument('kind',choices=['and','or']);args=p.parse_args()
r=Path(__file__).resolve().parent;prefix='logical-'+args.kind;op='Logical'+args.kind.title()+'FwdOp'
labels=[prefix+s for s in ['-boundaries','-manifest-v1','-public','-truth-edges']]
for label in labels:
 g=json.loads((r/'requests'/label/'guard.json').read_text());assert g['exit_code']==0 and g['performance_valid'],label
rows=[json.loads(l) for l in (r/(prefix+'-manifest-v1.jsonl')).read_text().splitlines()]
pairs=[v for v in json.loads((r/'baseline-scope.json').read_text())['pairs'] if v['op']==op]
assert len(rows)==18 and {v['case_id'] for v in rows}=={v['case'] for v in pairs}
for row in rows:
 assert row['correct'] and row['input_unchanged'] and len(row['latency_samples_ms'])==15 and len(row['eager']['latency_samples_ms'])==15
 assert row['runtime_limit']=={'cube_core_num':32,'vector_core_num':64}
 for path,digest in row['source_manifest'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
bound=json.loads((r/(prefix+'-boundaries.json')).read_text());assert bound['passed']==81
edges=json.loads((r/(prefix+'-truth-edges.json')).read_text());assert len(edges['checks'])==8 and all(v['passed'] for v in edges['checks'])
suites=list(ET.parse(r/(prefix+'-public.xml')).getroot().iter('testsuite'));assert sum(int(v.get('tests',0)) for v in suites)==({'and':13,'or':12}[args.kind]);assert all(int(v.get('failures',0))+int(v.get('errors',0))==0 for v in suites)
ratios=[v['eager']['latency_ms']/v['latency_ms'] for v in rows];assert min(ratios)>=.8
report={'op':op,'completed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'manifest_correct':len(rows),'same_host_eager_80_percent_passes':len(rows),'minimum_eager_ratio':min(ratios),'boundary_checks':81,'public_tests':{'and':13,'or':12}[args.kind],'truth_edge_checks':8,'runtime_limit':{'cube':32,'vector':64},'registered_baseline_pairs':len(pairs),'full_registered_baseline_acceptance':False,'source_manifest':rows[0]['source_manifest'],'guard_requests':labels,'measurement':prefix+'-manifest-v1.jsonl'}
(r/(prefix+'-REPORT.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
p=r.parent/'elementwise_serial_20261030/STATE.json';state=json.loads(p.read_text())
for v in state['operators']:
 if v['op']==op:v.update(local32_validated=True,status='baseline_pending',detail=f"本地32AIC：18/18正确且同机eager全部达80%，最差{min(ratios):.3f}；81边界+{report['public_tests']}公开+8位模式检查通过，登记baseline待补",evidence=str(r/(prefix+'-REPORT.json')))
state['updated_at']=report['completed_at'];t=p.with_suffix('.tmp');t.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n');t.replace(p)
print(json.dumps(report,ensure_ascii=False,indent=2))
