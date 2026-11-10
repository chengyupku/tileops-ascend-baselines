"""Audit all PReLU local32 evidence against the frozen production source."""
import json,hashlib,datetime
from pathlib import Path
import xml.etree.ElementTree as ET
r=Path(__file__).resolve().parent;op='PreluFwdOp'
for suffix in ['numerical-final-v2','boundaries-v2','empty-channels','tail-stress','public-final','manifest-v7']:
 g=json.loads((r/'requests'/('prelu-'+suffix)/'guard.json').read_text())
 assert g['exit_code']==0 and g['performance_valid'],suffix
rows=[json.loads(l) for l in (r/'prelu-manifest-v7.jsonl').read_text().splitlines()]
pairs=[v for v in json.loads((r/'baseline-scope.json').read_text())['pairs'] if v['op']==op]
assert len(rows)==4 and {v['case_id'] for v in rows}=={v['case'] for v in pairs}
for v in rows:
 assert v['correct'] and v['input_unchanged'] and v['verification']['checked_outputs']
 assert len(v['latency_samples_ms'])==len(v['eager']['latency_samples_ms'])==15
 assert v['runtime_limit']=={'cube_core_num':32,'vector_core_num':64}
 for path,h in v['source_manifest'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
module=r.parents[3]/'TileOPs/src/tileops/kernels/ascend/elementwise/prelu.py';h=hashlib.sha256(module.read_bytes()).hexdigest()
for suffix,count,key in [('boundaries',48,'rows'),('numerical-final',48,'checks'),('tail-stress',9,'checks')]:
 d=json.loads((r/('prelu-'+suffix+'.json')).read_text());assert d['module_sha256']==h
 assert d['runtime_limit']=={'cube_core_num':32,'vector_core_num':64}
 assert len(d[key])==count and all(v.get('correct',v.get('passed',False)) for v in d[key])
 if suffix=='boundaries':
  assert all(v['input_unchanged'] and v['changed_weight'] for v in d[key] if 'shape' in v)
  assert sum(v.get('weight_count')==8193 for v in d[key])==3
  assert sum(v.get('weight_count')==0 for v in d[key])==3
 elif suffix=='numerical-final':
  assert all(v['input_unchanged'] and v['signed_zero'] for v in d[key])
  for dtype in ['torch.float16','torch.bfloat16','torch.float32']:
   for layout,n in [('generic_tail',65537),('channel_simd',65536)]:
    subset=[v for v in d[key] if v['dtype']==dtype and v['layout']==layout]
    assert len(subset)==8 and all(v['values']==n for v in subset)
    assert {v['slope'] for v in subset}=={'0.0','-0.0','0.1','-1.0','2.0','inf','-inf','nan'}
 else:assert all(v['changed_inputs'] and v['input_unchanged'] for v in d[key])
empty=json.loads((r/'prelu-empty-channels.json').read_text());assert empty['module_sha256']==h and len(empty['checks'])==9 and all(v['passed'] for v in empty['checks'])
assert empty['runtime_limit']=={'cube_core_num':32,'vector_core_num':64}
suites=list(ET.parse(r/'prelu-public.xml').getroot().iter('testsuite'))
assert sum(int(s.get('tests',0)) for s in suites)==5
assert all(int(s.get('failures',0))+int(s.get('errors',0))+int(s.get('skipped',0))==0 for s in suites)
ratios=[v['eager']['latency_ms']/v['latency_ms'] for v in rows];assert min(ratios)>=.8
report={'op':op,'completed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'manifest_correct':4,'same_host_eager_80_percent_passes':4,'minimum_eager_ratio':min(ratios),'boundary_checks':48,'empty_channel_checks':9,'public_tests':5,'large_tail_checks':9,'numerical_checks':48,'numerical_scope':'all FP16/BF16 patterns plus FP32 random bits on generic and SIMD paths, eight finite/nonfinite slopes, signed zeros','runtime_limit':{'cube':32,'vector':64},'registered_baseline_pairs':len(pairs),'full_registered_baseline_acceptance':False,'source_manifest':rows[0]['source_manifest']}
(r/'prelu-REPORT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
p=r.parent/'elementwise_serial_20261030/STATE.json';s=json.loads(p.read_text())
for v in s['operators']:
 if v['op']==op:v.update(local32_validated=True,status='baseline_pending',detail=f"本地32AIC：4/4正确且eager达80%，最差{min(ratios):.3f}；48边界+5公开+9大尾部+48数值检查通过，baseline待补",evidence=str(r/'prelu-REPORT.json'))
s['updated_at']=report['completed_at'];tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
print(json.dumps(report,ensure_ascii=False,indent=2))
