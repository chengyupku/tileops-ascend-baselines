"""Audit Add final-source correctness and same-host limited-core performance."""
import json,hashlib,datetime
from pathlib import Path
import xml.etree.ElementTree as ET
r=Path(__file__).resolve().parent;op='AddFwdOp'
for label in ['add-bool-regression-v1','add-public-v2','add-broadcast-regression-v2','add-boundaries-v2','add-numerical-v2','add-tail-stress-v1','add-manifest-final-v1']:
 g=json.loads((r/'requests'/label/'guard.json').read_text());assert g['exit_code']==0 and g['performance_valid'],label
rows=[json.loads(l) for l in (r/'add-manifest-final-v1.jsonl').read_text().splitlines()]
pairs=[v for v in json.loads((r/'baseline-scope.json').read_text())['pairs'] if v['op']==op]
assert len(rows)==12 and {v['case_id'] for v in rows}=={v['case'] for v in pairs}
for v in rows:
 assert v['correct'] and v['input_unchanged'] and v['verification']['checked_outputs']
 assert len(v['latency_samples_ms'])==len(v['eager']['latency_samples_ms'])==15
 assert v['runtime_limit']=={'cube_core_num':32,'vector_core_num':64}
 for path,h in v['source_manifest'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
h=hashlib.sha256((r.parents[3]/'TileOPs/src/tileops/kernels/ascend/elementwise/arithmetic.py').read_bytes()).hexdigest()
boolean=json.loads((r/'add-bool-regression.json').read_text())
assert boolean['module_sha256']==h and boolean['runtime_limit']=={'cube_core_num':32,'vector_core_num':64}
assert len(boolean['checks'])==3 and {v['alpha'] for v in boolean['checks']}=={0,1,2}
assert all(v['pairs']==65536 and v['cpu_mismatches']==v['npu_mismatches']==0 for v in boolean['checks'])
dtypes={'torch.'+s for s in ['bool','uint8','int8','int16','int32','int64','float16','bfloat16','float32']};floats={'torch.float16','torch.bfloat16','torch.float32'}
shapes={((),()),((0,3),(1,3)),((65,),(65,)),((129,),()),((2,1,5),(1,3,1)),((1,5),(3,1)),((3,5),(1,5)),((16387,),(16387,)),((2,4,64),(4,1)),((2,3,65),(3,1))}
for suffix,n in [('broadcast-regression',9),('boundaries',90),('numerical',30),('tail-stress',30)]:
 d=json.loads((r/('add-'+suffix+'.json')).read_text());assert d['module_sha256']==h and d['runtime_limit']=={'cube_core_num':32,'vector_core_num':64}
 checks=d['checks'];assert len(checks)==n and all(v['passed'] for v in checks)
 expected=floats if suffix=='broadcast-regression' else dtypes
 assert {v['dtype'] for v in checks}==expected
 for dtype in expected:
  subset=[v for v in checks if v['dtype']==dtype]
  if suffix=='broadcast-regression':
   assert len(subset)==3 and {v['period'] for v in subset}=={960,1088,2112}
   assert all(v['rows']==5 and v['alpha']==.5 and v['sentinel'] and v['input_unchanged'] for v in subset)
  elif suffix=='boundaries':
   assert len(subset)==10 and {(tuple(v['a_shape']),tuple(v['b_shape'])) for v in subset}==shapes
   assert all(v['alpha_changed'] and v['input_unchanged'] for v in subset)
  elif suffix=='numerical':
   alphas={0,1,2} if dtype in {'torch.bool','torch.uint8'} else {1,-1,.5,2} if dtype in floats else {1,-1,2}
   assert len(subset)==len(alphas) and {v['alpha'] for v in subset}==alphas
   assert all(v['values']==65537 and v['input_unchanged'] for v in subset)
  else:
   tails=[v for v in subset if 'check' not in v];assert len(tails)==3 and {v['n'] for v in tails}=={32767,524289,540671}
   assert all(v['changed_inputs'] and v['input_unchanged'] for v in tails)
   sentinels=[v for v in subset if v.get('check')=='sentinel_and_negative_control'];assert len(sentinels)==int(dtype in floats)
suites=list(ET.parse(r/'add-public.xml').getroot().iter('testsuite'));assert sum(int(s.get('tests',0)) for s in suites)==24
assert all(int(s.get('failures',0))+int(s.get('errors',0))+int(s.get('skipped',0))==0 for s in suites)
ratio=min(v['eager']['latency_ms']/v['latency_ms'] for v in rows);assert ratio>=.8
report={'op':op,'completed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'manifest_correct':12,'same_host_eager_80_percent_passes':12,'minimum_eager_ratio':ratio,'public_tests':24,'broadcast_regressions':9,'raw_bool_pair_checks':196608,'boundary_checks':90,'numerical_checks':30,'tail_and_sentinel_checks':30,'runtime_limit':{'cube':32,'vector':64},'registered_baseline_pairs':len(pairs),'full_registered_baseline_acceptance':False,'source_manifest':rows[0]['source_manifest']}
(r/'add-REPORT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
p=r.parent/'elementwise_serial_20261030/STATE.json';s=json.loads(p.read_text())
for v in s['operators']:
 if v['op']==op:v.update(local32_validated=True,status='baseline_pending',detail=f'本地32AIC：12/12正确且eager达80%，最差{ratio:.3f}；24公开+9广播回归+90边界+30数值+30尾块/哨兵通过，baseline待补',evidence=str(r/'add-REPORT.json'))
s['updated_at']=report['completed_at'];tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
print(json.dumps(report,ensure_ascii=False,indent=2))
