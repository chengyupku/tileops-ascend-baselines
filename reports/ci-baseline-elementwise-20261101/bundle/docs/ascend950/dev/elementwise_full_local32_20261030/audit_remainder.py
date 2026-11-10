"""Audit Remainder final-source correctness and same-host limited-core performance."""
import json,hashlib,datetime
from pathlib import Path
import xml.etree.ElementTree as ET
r=Path(__file__).resolve().parent;op='RemainderFwdOp'
for label in ['remainder-public-v1','remainder-numerical-v1','remainder-manifest-v2','remainder-broadcast-regression-v2','remainder-boundaries-v2','remainder-tail-stress-v2','remainder-ub-regression-v2']:
 g=json.loads((r/'requests'/label/'guard.json').read_text());assert g['exit_code']==0 and g['performance_valid'],label
rows=[json.loads(l) for l in (r/'remainder-manifest-v2.jsonl').read_text().splitlines()]
pairs=[v for v in json.loads((r/'baseline-scope.json').read_text())['pairs'] if v['op']==op]
assert len(rows)==14 and {v['case_id'] for v in rows}=={v['case'] for v in pairs}
for v in rows:
 assert v['correct'] and v['input_unchanged'] and v['verification']['checked_outputs']
 assert len(v['latency_samples_ms'])==len(v['eager']['latency_samples_ms'])==15
 assert v['runtime_limit']=={'cube_core_num':32,'vector_core_num':64}
 for path,h in v['source_manifest'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
h=hashlib.sha256((r.parents[3]/'TileOPs/src/tileops/kernels/ascend/elementwise/arithmetic_modulo.py').read_bytes()).hexdigest()
rh=hashlib.sha256((r/'remainder_reference.py').read_bytes()).hexdigest()
dtypes=floats={'torch.float16','torch.bfloat16','torch.float32'}
shapes={((),()),((0,3),(1,3)),((65,),(65,)),((129,),()),((2,1,5),(1,3,1)),((1,5),(3,1)),((3,5),(1,5)),((16387,),(16387,)),((2,4,64),(4,1)),((2,3,65),(3,1))}
for suffix,n in [('broadcast-regression',18),('boundaries',30),('numerical',9),('tail-stress',12),('ub-regression',12)]:
 d=json.loads((r/('remainder-'+suffix+'.json')).read_text());assert d['module_sha256']==h and d['runtime_limit']=={'cube_core_num':32,'vector_core_num':64}
 assert d['reference_sha256']==rh
 checks=d['checks'];assert len(checks)==n and all(v['passed'] for v in checks)
 expected=floats if suffix=='broadcast-regression' else dtypes
 assert {v['dtype'] for v in checks}==expected
 for dtype in expected:
  subset=[v for v in checks if v['dtype']==dtype]
  if suffix=='ub-regression':
   assert len(subset)==4 and {(tuple(v['a_shape']),tuple(v['b_shape'])) for v in subset}=={((2,11008),(1,11008)),((2,16320),(1,16320)),((2,16384),(1,16384)),((1,4,8192),(4,1))}
   assert all(v['input_unchanged'] for v in subset)
  elif suffix=='broadcast-regression':
   assert len(subset)==6 and {(v['period'],v['rows']) for v in subset}=={(p,n) for p in [960,1088,2112] for n in [5,6]}
   assert all(v['sentinel'] and v['input_unchanged'] for v in subset)
  elif suffix=='boundaries':
   assert len(subset)==10 and {(tuple(v['a_shape']),tuple(v['b_shape'])) for v in subset}==shapes
   assert all(v['changed_inputs'] and v['input_unchanged'] for v in subset)
  elif suffix=='numerical':
   assert len(subset)==3 and {v['pattern'] for v in subset}=={'rolled','reversed','square'}
   assert all(v['values']==65537 and v['input_unchanged'] for v in subset)
  else:
   tails=[v for v in subset if 'check' not in v];assert len(tails)==3 and {v['n'] for v in tails}=={32767,524289,540671}
   assert all(v['changed_inputs'] and v['input_unchanged'] for v in tails)
   sentinels=[v for v in subset if v.get('check')=='sentinel_and_negative_control'];assert len(sentinels)==int(dtype in floats)
for path,expected in json.loads((r/'remainder-public-v1-sources.json').read_text())['source_manifest'].items():
 assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,path
suites=list(ET.parse(r/'remainder-public.xml').getroot().iter('testsuite'));assert sum(int(s.get('tests',0)) for s in suites)==20
assert all(int(s.get('failures',0))+int(s.get('errors',0))+int(s.get('skipped',0))==0 for s in suites)
ratio=min(v['eager']['latency_ms']/v['latency_ms'] for v in rows);assert ratio>=.8
report={'op':op,'completed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'manifest_correct':14,'same_host_eager_80_percent_passes':14,'minimum_eager_ratio':ratio,'public_tests':20,'broadcast_regressions':18,'boundary_checks':30,'numerical_checks':9,'tail_and_sentinel_checks':12,'ub_regressions':12,'runtime_limit':{'cube':32,'vector':64},'registered_baseline_pairs':len(pairs),'full_registered_baseline_acceptance':False,'source_manifest':rows[0]['source_manifest'],'reference_sha256':rh,'reference_note':'Original CUDA FP32 fmod followed by nonzero sign correction and one narrowing. Scalar libm avoids demonstrated CPU vector remainder/fmod extreme-ratio errors; original zero-tolerance contract preserved. Original public tests unchanged for Remainder.'}
(r/'remainder-REPORT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
p=r.parent/'elementwise_serial_20261030/STATE.json';s=json.loads(p.read_text())
for v in s['operators']:
 if v['op']==op:v.update(local32_validated=True,status='baseline_pending',detail=f'本地32AIC：14/14正确且eager达80%，最差{ratio:.3f}；20公开+18广播+30边界+9数值+12尾块+12UB通过，baseline待补',evidence=str(r/'remainder-REPORT.json'))
s['updated_at']=report['completed_at'];tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
print(json.dumps(report,ensure_ascii=False,indent=2))
