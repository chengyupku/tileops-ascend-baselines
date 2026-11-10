"""Audit Dropout local32 results; missing baseline coverage is never acceptance."""
import json,hashlib,datetime
from pathlib import Path
import xml.etree.ElementTree as ET
r=Path(__file__).resolve().parent;op='DropoutFwdOp'
for suffix in ['public','boundaries','tail-stress','manifest-v1']:
 g=json.loads((r/'requests'/('dropout-'+suffix)/'guard.json').read_text())
 assert g['exit_code']==0 and g['performance_valid'],suffix
rows=[json.loads(l) for l in (r/'dropout-manifest-v1.jsonl').read_text().splitlines()]
assert len(rows)==3 and {v['case_id'] for v in rows}=={'hidden-4k-float16','hidden-4k-float32','hidden-10k-bfloat16'}
for v in rows:
 assert v['correct'] and v['input_unchanged'] and v['verification']['checked_outputs']
 assert len(v['latency_samples_ms'])==len(v['eager']['latency_samples_ms'])==15
 assert v['runtime_limit']=={'cube_core_num':32,'vector_core_num':64}
 for path,h in v['source_manifest'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
module=r.parents[3]/'TileOPs/src/tileops/kernels/ascend/elementwise/dropout.py';h=hashlib.sha256(module.read_bytes()).hexdigest()
b=json.loads((r/'dropout-boundaries.json').read_text());assert b['module_sha256']==h
assert b['runtime_limit']=={'cube_core_num':32,'vector_core_num':64}
assert len(b['rows'])==36 and all(v['correct'] for v in b['rows'])
assert all(v['input_unchanged'] and v['replay'] for v in b['rows'] if 'shape' in v)
t=json.loads((r/'dropout-tail-stress.json').read_text());assert t['module_sha256']==h
assert t['runtime_limit']=={'cube_core_num':32,'vector_core_num':64}
assert len(t['checks'])==12 and all(v['passed'] and v['replay'] and v['changed_input'] and v['cross_tile_checked'] for v in t['checks'])
assert sum(v['tail_collision_checked'] for v in t['checks'])==3
suites=list(ET.parse(r/'dropout-public.xml').getroot().iter('testsuite'))
assert sum(int(s.get('tests',0)) for s in suites)==22
assert all(int(s.get('failures',0))+int(s.get('errors',0))+int(s.get('skipped',0))==0 for s in suites)
ratios=[v['eager']['latency_ms']/v['latency_ms'] for v in rows];assert min(ratios)>=.8
pairs=[v for v in json.loads((r/'baseline-scope.json').read_text())['pairs'] if v['op']==op]
report={'op':op,'completed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'manifest_correct':3,'same_host_eager_80_percent_passes':3,'minimum_eager_ratio':min(ratios),'boundary_checks':36,'public_tests':22,'large_tail_and_rng_checks':12,'runtime_limit':{'cube':32,'vector':64},'registered_baseline_pairs':len(pairs),'baseline_coverage_missing':not bool(pairs),'full_registered_baseline_acceptance':False,'source_manifest':rows[0]['source_manifest']}
(r/'dropout-REPORT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
p=r.parent/'elementwise_serial_20261030/STATE.json';s=json.loads(p.read_text())
for v in s['operators']:
 if v['op']==op:v.update(local32_validated=True,status='baseline_pending',detail=f"本地32AIC：3/3正确且eager达80%，最差{min(ratios):.3f}；36边界+22公开+12大尾部/RNG通过，baseline覆盖待补证",evidence=str(r/'dropout-REPORT.json'))
s['updated_at']=report['completed_at'];tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
print(json.dumps(report,ensure_ascii=False,indent=2))
