"""Audit final Tanh kernel under the local32 runtime limit."""
import argparse,json,hashlib,datetime
from pathlib import Path
import xml.etree.ElementTree as ET
from types import SimpleNamespace
a=SimpleNamespace(kind='tanh')
r=Path(__file__).resolve().parent;op='TanhFwdOp'
for suffix in ['boundaries','tail-stress','numerical-final','public','manifest-v1']:
 label=a.kind+'-'+suffix;g=json.loads((r/'requests'/label/'guard.json').read_text());assert g['exit_code']==0 and g['performance_valid'],label
rows=[json.loads(l) for l in (r/(a.kind+'-manifest-v1.jsonl')).read_text().splitlines()]
pairs=[v for v in json.loads((r/'baseline-scope.json').read_text())['pairs'] if v['op']==op]
assert len(rows)==5 and {v['case_id'] for v in rows}=={v['case'] for v in pairs}
for v in rows:
 assert v['correct'] and v['input_unchanged'] and v['verification']['checked_outputs']
 assert len(v['latency_samples_ms'])==len(v['eager']['latency_samples_ms'])==15
 assert v['runtime_limit']=={'cube_core_num':32,'vector_core_num':64}
 for path,h in v['source_manifest'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
module=r.parents[3]/('TileOPs/src/tileops/kernels/ascend/elementwise/'+a.kind+'.py');h=hashlib.sha256(module.read_bytes()).hexdigest()
for suffix,n in [('tail-stress',9),('numerical-final',3)]:
 d=json.loads((r/(a.kind+'-'+suffix+'.json')).read_text());assert len(d['checks'])==n and all(v['passed'] for v in d['checks']);assert d['module_sha256']==h
 if suffix=='tail-stress':assert all(v['changed_input'] for v in d['checks'])
 else:assert [v['values'] for v in d['checks']]==[65537,65537,65564]
b=json.loads((r/(a.kind+'-boundaries.json')).read_text());assert len(b['rows'])==36 and all(v['correct'] for v in b['rows']);assert all(v['input_unchanged'] for v in b['rows'] if 'shape' in v)
for path,h in b['sources'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
suites=list(ET.parse(r/(a.kind+'-public.xml')).getroot().iter('testsuite'));n_public=sum(int(s.get('tests',0)) for s in suites)
assert n_public==4 and all(int(s.get('errors',0))+int(s.get('failures',0))+int(s.get('skipped',0))==0 for s in suites)
ratios=[v['eager']['latency_ms']/v['latency_ms'] for v in rows];assert min(ratios)>=.8
report={'op':op,'completed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'manifest_correct':5,'same_host_eager_80_percent_passes':5,'minimum_eager_ratio':min(ratios),'boundary_checks':36,'public_tests':n_public,'large_tail_checks':9,'numerical_checks':'CPU FP64 oracle; all FP16/BF16 patterns plus tail; 65564 FP32 values, original numerical contract','runtime_limit':{'cube':32,'vector':64},'registered_baseline_pairs':len(pairs),'full_registered_baseline_acceptance':False,'source_manifest':rows[0]['source_manifest']}
(r/(a.kind+'-REPORT.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
p=r.parent/'elementwise_serial_20261030/STATE.json';s=json.loads(p.read_text())
for v in s['operators']:
 if v['op']==op:v.update(local32_validated=True,status='baseline_pending',detail=f"本地32AIC：5/5正确且同机eager达80%，最差{min(ratios):.3f}；36边界+4公开+9大尾部+全位模式通过，登记baseline待补",evidence=str(r/(a.kind+'-REPORT.json')))
s['updated_at']=report['completed_at'];t=p.with_suffix('.tmp');t.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n');t.replace(p)
print(json.dumps(report,ensure_ascii=False,indent=2))
