"""Audit FloorDivide final-source correctness and same-host limited-core performance."""
import json,hashlib,datetime
from pathlib import Path
import xml.etree.ElementTree as ET
r=Path(__file__).resolve().parent;op='FloorDivideFwdOp'
for label in ['floor-divide-public-v2','floor-divide-numerical-v2','floor-divide-manifest-v2','floor-divide-broadcast-regression-v1','floor-divide-boundaries-v1','floor-divide-tail-stress-v1','floor-divide-ub-regression-v2','floor-divide-shared-div-public-v1','floor-divide-shared-remainder-public-v1','floor-divide-fast-boundaries-v2','floor-divide-repair-v1']:
 g=json.loads((r/'requests'/label/'guard.json').read_text());assert g['exit_code']==0 and g['performance_valid'],label
rows=[json.loads(l) for l in (r/'floor_divide-manifest-v2.jsonl').read_text().splitlines()]
pairs=[v for v in json.loads((r/'baseline-scope.json').read_text())['pairs'] if v['op']==op]
assert len(rows)==14 and {v['case_id'] for v in rows}=={v['case'] for v in pairs}
for v in rows:
 assert v['correct'] and v['input_unchanged'] and v['verification']['checked_outputs']
 assert len(v['latency_samples_ms'])==len(v['eager']['latency_samples_ms'])==15
 assert v['runtime_limit']=={'cube_core_num':32,'vector_core_num':64}
 for path,h in v['source_manifest'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
h=hashlib.sha256((r.parents[3]/'TileOPs/src/tileops/kernels/ascend/elementwise/arithmetic_modulo.py').read_bytes()).hexdigest()
rh=hashlib.sha256((r/'floor_divide_reference.py').read_bytes()).hexdigest()
dtypes=floats={'torch.float16','torch.bfloat16','torch.float32'}
shapes={((),()),((0,3),(1,3)),((65,),(65,)),((129,),()),((2,1,5),(1,3,1)),((1,5),(3,1)),((3,5),(1,5)),((16387,),(16387,)),((2,4,64),(4,1)),((2,3,65),(3,1))}
for suffix,n in [('broadcast-regression',18),('boundaries',30),('numerical',9),('tail-stress',12),('ub-regression',12),('fast-boundaries',27),('repair',12)]:
 d=json.loads((r/('floor_divide-'+suffix+'.json')).read_text());assert d['module_sha256']==h and d['runtime_limit']=={'cube_core_num':32,'vector_core_num':64}
 assert d['reference_sha256']==rh
 assert len(d['dependency_manifest'])==3
 for path,expected_hash in d['dependency_manifest'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected_hash,path
 checks=d['checks'];assert len(checks)==n and all(v['passed'] for v in checks)
 expected=floats if suffix=='broadcast-regression' else dtypes
 assert {v['dtype'] for v in checks}==expected
 for dtype in expected:
  subset=[v for v in checks if v['dtype']==dtype]
  if suffix=='fast-boundaries':
   assert len(subset)==9 and {(v['layout'],v['neighbor']) for v in subset}=={(layout,shift) for layout in ['same','bias','channel'] for shift in [-1,0,1]}
   assert all(v['input_unchanged'] and v['values']==(61440 if v['layout']=='bias' else 65536) for v in subset)
  elif suffix=='repair':
   assert len(subset)==4 and {(tuple(v['a_shape']),tuple(v['b_shape'])) for v in subset}=={((65537,),(65537,)),((5,960),(1,960)),((6,1088),(1,1088)),((2,64,128),(64,1))}
   assert all(v['input_unchanged'] and v['all_quotients_above_fast_limit'] for v in subset)
  elif suffix=='ub-regression':
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
for path,expected in json.loads((r/'floor-divide-public-v2-sources.json').read_text())['source_manifest'].items():
 assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,path
suites=list(ET.parse(r/'floor_divide-public.xml').getroot().iter('testsuite'));assert sum(int(s.get('tests',0)) for s in suites)==20
assert all(int(s.get('failures',0))+int(s.get('errors',0))+int(s.get('skipped',0))==0 for s in suites)
import ast
scope=json.loads((r/'floor-divide-shared-source-scope.json').read_text())
before=Path(scope['prior_source']).read_text()
after=(r.parents[3]/'TileOPs/src/tileops/kernels/ascend/elementwise/arithmetic_modulo.py').read_text()
assert hashlib.sha256(before.encode()).hexdigest()==scope['before_sha256']
assert hashlib.sha256(after.encode()).hexdigest()==scope['after_sha256']
def unaffected(source):
 tree=ast.parse(source);tree.body=[v for v in tree.body if not (isinstance(v,ast.ClassDef) and v.name=='AscendFloorDivideKernel')]
 return ast.dump(tree,include_attributes=False)
assert unaffected(before)==unaffected(after)
for filename,count in [('div-public.xml',32),('remainder-public.xml',20)]:
 suites=list(ET.parse(r/filename).getroot().iter('testsuite'))
 assert sum(int(s.get('tests',0)) for s in suites)==count
 assert all(int(s.get('failures',0))+int(s.get('errors',0))+int(s.get('skipped',0))==0 for s in suites)
ratio=min(v['eager']['latency_ms']/v['latency_ms'] for v in rows);assert ratio>=.8
report={'op':op,'completed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'manifest_correct':14,'same_host_eager_80_percent_passes':14,'minimum_eager_ratio':ratio,'public_tests':20,'broadcast_regressions':18,'boundary_checks':30,'numerical_checks':9,'tail_and_sentinel_checks':12,'ub_regressions':12,'fast_integer_boundary_checks':27,'forced_repair_checks':12,'runtime_limit':{'cube':32,'vector':64},'shared_div_public_regressions':32,'shared_remainder_public_regressions':20,'shared_source_scope':scope,'registered_baseline_pairs':len(pairs),'full_registered_baseline_acceptance':False,'source_manifest':rows[0]['source_manifest'],'reference_sha256':rh,'reference_note':'Original CUDA FloorDivide floor tiers, with scalar libm fmod and input-dtype rounding. Original zero-tolerance contract and bitwise assertions preserved. NPU public references use the same independently verified Div floor oracle.'}
(r/'floor_divide-REPORT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
p=r.parent/'elementwise_serial_20261030/STATE.json';s=json.loads(p.read_text())
for v in s['operators']:
 if v['op']==op:v.update(local32_validated=True,status='baseline_pending',detail=f'本地32AIC：14/14正确且eager达80%，最差{ratio:.3f}；20公开+18广播+30边界+9数值+12尾块+12UB+27快速边界+12强制修正通过，baseline待补',evidence=str(r/'floor_divide-REPORT.json'))
s['updated_at']=report['completed_at'];tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
print(json.dumps(report,ensure_ascii=False,indent=2))
