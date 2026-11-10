"""Audit Sub final-source correctness and same-host limited-core performance."""
import json,hashlib,datetime
from pathlib import Path
import xml.etree.ElementTree as ET
r=Path(__file__).resolve().parent;op='SubFwdOp'
for label in ['sub-alpha-extremes-v1','sub-public-v1','sub-broadcast-regression-v1','sub-boundaries-v1','sub-numerical-v1','sub-tail-stress-v1','sub-manifest-v1']:
 g=json.loads((r/'requests'/label/'guard.json').read_text());assert g['exit_code']==0 and g['performance_valid'],label
rows=[json.loads(l) for l in (r/'sub-manifest-v1.jsonl').read_text().splitlines()]
pairs=[v for v in json.loads((r/'baseline-scope.json').read_text())['pairs'] if v['op']==op]
assert len(rows)==16 and {v['case_id'] for v in rows}=={v['case'] for v in pairs}
for v in rows:
 assert v['correct'] and v['input_unchanged'] and v['verification']['checked_outputs']
 assert len(v['latency_samples_ms'])==len(v['eager']['latency_samples_ms'])==15
 assert v['runtime_limit']=={'cube_core_num':32,'vector_core_num':64}
 for path,h in v['source_manifest'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
h=hashlib.sha256((r.parents[3]/'TileOPs/src/tileops/kernels/ascend/elementwise/arithmetic_sub.py').read_bytes()).hexdigest()
extremes=json.loads((r/'sub-alpha-extremes.json').read_text())
assert extremes['module_sha256']==h and extremes['runtime_limit']=={'cube_core_num':32,'vector_core_num':64}
assert len(extremes['checks'])==24 and all(v['passed'] and v['input_unchanged'] and v['values']==129 for v in extremes['checks'])
for dtype,bits in [('torch.uint8',8),('torch.int8',8),('torch.int16',16),('torch.int32',32),('torch.int64',64)]:
 subset=[v for v in extremes['checks'] if v['dtype']==dtype]
 alphas={0,1,2,255} if dtype=='torch.uint8' else {0,1,-1,-(1<<(bits-1)),(1<<(bits-1))-1}
 assert len(subset)==len(alphas) and {v['alpha'] for v in subset}==alphas
 for v in subset:
  if dtype in {'torch.int8','torch.int16','torch.int32'} and v['alpha']==-(1<<(bits-1)):
   assert 'overflow' in v['cpu_reference_error']
  else:assert v['cpu_reference_agrees']
dtypes={'torch.'+s for s in ['uint8','int8','int16','int32','int64','float16','bfloat16','float32']};floats={'torch.float16','torch.bfloat16','torch.float32'}
shapes={((),()),((0,3),(1,3)),((65,),(65,)),((129,),()),((2,1,5),(1,3,1)),((1,5),(3,1)),((3,5),(1,5)),((16387,),(16387,)),((2,4,64),(4,1)),((2,3,65),(3,1))}
for suffix,n in [('broadcast-regression',18),('boundaries',80),('numerical',27),('tail-stress',27)]:
 d=json.loads((r/('sub-'+suffix+'.json')).read_text());assert d['module_sha256']==h and d['runtime_limit']=={'cube_core_num':32,'vector_core_num':64}
 checks=d['checks'];assert len(checks)==n and all(v['passed'] for v in checks)
 expected=floats if suffix=='broadcast-regression' else dtypes
 assert {v['dtype'] for v in checks}==expected
 for dtype in expected:
  subset=[v for v in checks if v['dtype']==dtype]
  if suffix=='broadcast-regression':
   assert len(subset)==6 and {(v['period'],v['rows']) for v in subset}=={(p,n) for p in [960,1088,2112] for n in [5,6]}
   assert all(v['alpha']==.5 and v['sentinel'] and v['input_unchanged'] for v in subset)
  elif suffix=='boundaries':
   assert len(subset)==10 and {(tuple(v['a_shape']),tuple(v['b_shape'])) for v in subset}==shapes
   assert all(v['alpha_changed'] and v['input_unchanged'] for v in subset)
  elif suffix=='numerical':
   alphas={0,1,2} if dtype=='torch.uint8' else {1,-1,.5,2} if dtype in floats else {1,-1,2}
   assert len(subset)==len(alphas) and {v['alpha'] for v in subset}==alphas
   assert all(v['values']==65537 and v['input_unchanged'] for v in subset)
  else:
   tails=[v for v in subset if 'check' not in v];assert len(tails)==3 and {v['n'] for v in tails}=={32767,524289,540671}
   assert all(v['changed_inputs'] and v['input_unchanged'] for v in tails)
   sentinels=[v for v in subset if v.get('check')=='sentinel_and_negative_control'];assert len(sentinels)==int(dtype in floats)
suites=list(ET.parse(r/'sub-public.xml').getroot().iter('testsuite'));assert sum(int(s.get('tests',0)) for s in suites)==11
assert all(int(s.get('failures',0))+int(s.get('errors',0))+int(s.get('skipped',0))==0 for s in suites)
ratio=min(v['eager']['latency_ms']/v['latency_ms'] for v in rows);assert ratio>=.8
report={'op':op,'completed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'manifest_correct':16,'same_host_eager_80_percent_passes':16,'minimum_eager_ratio':ratio,'public_tests':11,'broadcast_regressions':18,'integer_alpha_extremes':24,'boundary_checks':80,'numerical_checks':27,'tail_and_sentinel_checks':27,'runtime_limit':{'cube':32,'vector':64},'registered_baseline_pairs':len(pairs),'full_registered_baseline_acceptance':False,'source_manifest':rows[0]['source_manifest'],'reference_note':'For int8/int16/int32 minimum alpha, CPU torch.sub rejects a manifest-representable alpha; exact Python modular subtraction is checked for all 24 integer alpha cases, and must also agree with CPU torch.sub wherever the latter executes.'}
(r/'sub-REPORT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
p=r.parent/'elementwise_serial_20261030/STATE.json';s=json.loads(p.read_text())
for v in s['operators']:
 if v['op']==op:v.update(local32_validated=True,status='baseline_pending',detail=f'本地32AIC：16/16正确且eager达80%，最差{ratio:.3f}；11公开+18广播回归+80边界+27数值+24alpha极值+27尾块/哨兵通过，baseline待补',evidence=str(r/'sub-REPORT.json'))
s['updated_at']=report['completed_at'];tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
print(json.dumps(report,ensure_ascii=False,indent=2))
