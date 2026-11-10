"""Audit Minimum final-source correctness and same-host limited-core performance."""
import json,hashlib,datetime
from pathlib import Path
import xml.etree.ElementTree as ET
r=Path(__file__).resolve().parent;op='MinimumFwdOp'
for label in ['minimum-public-v1','minimum-shared-maximum-v1','minimum-numerical-v2','minimum-manifest-v2','minimum-broadcast-v1','minimum-boundaries-v1','minimum-tail-v1','minimum-ub-v1','minimum-special-v1','minimum-channel-v1','minimum-bool-probe-v2','minimum-bool-broadcast-v1']:
 g=json.loads((r/'requests'/label/'guard.json').read_text());assert g['exit_code']==0 and g['performance_valid'],label
rows=[json.loads(l) for l in (r/'minimum-manifest-v2.jsonl').read_text().splitlines()]
pairs=[v for v in json.loads((r/'baseline-scope.json').read_text())['pairs'] if v['op']==op]
assert len(rows)==14 and {v['case_id'] for v in rows}=={v['case'] for v in pairs}
for v in rows:
 assert v['correct'] and v['input_unchanged'] and v['verification']['checked_outputs']
 assert len(v['latency_samples_ms'])==len(v['eager']['latency_samples_ms'])==15
 assert v['runtime_limit']=={'cube_core_num':32,'vector_core_num':64}
 for path,h in v['source_manifest'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
h=hashlib.sha256((r.parents[3]/'TileOPs/src/tileops/kernels/ascend/elementwise/arithmetic_extrema.py').read_bytes()).hexdigest()
rh=hashlib.sha256((r/'minimum_reference.py').read_bytes()).hexdigest()
dtypes={'torch.'+s for s in ['bool','uint8','int8','int16','int32','int64','float16','bfloat16','float32']};floats={'torch.float16','torch.bfloat16','torch.float32'}
shapes={((),()),((0,3),(1,3)),((65,),(65,)),((129,),()),((2,1,5),(1,3,1)),((1,5),(3,1)),((3,5),(1,5)),((16387,),(16387,)),((2,4,64),(4,1)),((2,3,65),(3,1))}
for suffix,n in [('broadcast-regression',18),('boundaries',90),('numerical',27),('tail-stress',30),('ub-regression',12),('special-values',9),('channel-stress',18)]:
 d=json.loads((r/('minimum-'+suffix+'.json')).read_text());assert d['module_sha256']==h and d['runtime_limit']=={'cube_core_num':32,'vector_core_num':64}
 assert d['reference_sha256']==rh
 assert len(d['dependency_manifest'])==3
 for path,expected_hash in d['dependency_manifest'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected_hash,path
 checks=d['checks'];assert len(checks)==n and all(v['passed'] for v in checks)
 expected=floats if suffix in {'broadcast-regression','ub-regression','special-values','channel-stress'} else dtypes
 assert {v['dtype'] for v in checks}==expected
 for dtype in expected:
  subset=[v for v in checks if v['dtype']==dtype]
  if suffix=='channel-stress':
   assert len(subset)==6 and {(v['period'],v['repeat']) for v in subset}=={(p,n) for p in [3,19] for n in [64,65536,65600]}
   assert all(v['input_unchanged'] for v in subset)
  elif suffix=='special-values':
   assert len(subset)==3 and {v['layout'] for v in subset}=={'same','bias','channel'}
   assert all(v['pairs']==361 and v['signed_zero'] and v['input_unchanged'] for v in subset)
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
   assert len(subset)==3 and {v['mode'] for v in subset}=={'rolled','reversed','square'}
   assert all(v['values']==65537 and v['input_unchanged'] for v in subset)
  else:
   tails=[v for v in subset if 'check' not in v];assert len(tails)==3 and {v['n'] for v in tails}=={32767,524289,540671}
   assert all(v['changed_inputs'] and v['input_unchanged'] for v in tails)
   sentinels=[v for v in subset if v.get('check')=='sentinel_and_negative_control'];assert len(sentinels)==int(dtype in floats)
for path,expected in json.loads((r/'minimum-final-v2-sources.json').read_text())['source_manifest'].items():
 assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,path
suites=list(ET.parse(r/'minimum-public.xml').getroot().iter('testsuite'));assert sum(int(s.get('tests',0)) for s in suites)==24
assert all(int(s.get('failures',0))+int(s.get('errors',0))+int(s.get('skipped',0))==0 for s in suites)
suites=list(ET.parse(r/'maximum-public.xml').getroot().iter('testsuite'));assert sum(int(s.get('tests',0)) for s in suites)==26
assert all(int(s.get('failures',0))+int(s.get('errors',0))+int(s.get('skipped',0))==0 for s in suites)
import ast
scope=json.loads((r/'minimum-shared-source-scope.json').read_text())
before=Path(scope['prior_source']).read_text()
after=(r.parents[3]/'TileOPs/src/tileops/kernels/ascend/elementwise/arithmetic_extrema.py').read_text()
assert hashlib.sha256(before.encode()).hexdigest()==scope['before_sha256']
assert hashlib.sha256(after.encode()).hexdigest()==scope['after_sha256']==h
def unchanged_maximum(source):
 tree=ast.parse(source);tree.body=[v for v in tree.body if not (isinstance(v,ast.ClassDef) and v.name=='AscendMinimumKernel')]
 return ast.dump(tree,include_attributes=False)
assert unchanged_maximum(before)==unchanged_maximum(after)
maximum_report=json.loads((r/'maximum-REPORT.json').read_text())
for path,expected in maximum_report['source_manifest'].items():
 if Path(path).name=='arithmetic_extrema.py':assert expected==scope['before_sha256']
 else:assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,path
for suffix in ['bool-probe','bool-broadcast']:
 d=json.loads((r/('minimum-'+suffix+'.json')).read_text())
 assert d['module_sha256']==h and d['runtime_limit']=={'cube_core_num':32,'vector_core_num':64}
 assert len(d['dependency_manifest'])==3
 for path,expected in d['dependency_manifest'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected
 if suffix=='bool-probe':
  assert d['pairs']==65536 and d['op_mismatches']==0 and d['input_unchanged']
 else:
  checks=d['checks'];assert len(checks)==6
  assert {(v['layout'],v['swapped']) for v in checks}=={(layout,swapped) for layout in ['same','bias','channel'] for swapped in [False,True]}
  assert all(v['passed'] and v['pairs']==65536 and v['canonical_output'] and v['input_unchanged'] for v in checks)
ratio=min(v['eager']['latency_ms']/v['latency_ms'] for v in rows);assert ratio>=.8
report={'op':op,'completed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'manifest_correct':14,'same_host_eager_80_percent_passes':14,'minimum_eager_ratio':ratio,'public_tests':24,'broadcast_regressions':18,'boundary_checks':90,'numerical_checks':27,'tail_and_sentinel_checks':30,'ub_regressions':12,'special_value_checks':9,'channel_stress_checks':18,'public_evidence_request':'minimum-public-v1','shared_maximum_public_tests':26,'raw_bool_pairs':65536,'bool_broadcast_checks':6,'shared_source_scope':scope,'runtime_limit':{'cube':32,'vector':64},'registered_baseline_pairs':len(pairs),'full_registered_baseline_acceptance':False,'source_manifest':rows[0]['source_manifest'],'reference_sha256':rh,'reference_note':'CPU torch.minimum with original CUDA IEEE zero ties and NaN propagation; bool storage interpreted by nonzero truth. Exact zero-tolerance workload and original public assertions preserved.'}
(r/'minimum-REPORT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
p=r.parent/'elementwise_serial_20261030/STATE.json';s=json.loads(p.read_text())
for v in s['operators']:
 if v['op']==op:v.update(local32_validated=True,status='baseline_pending',detail=f'本地32AIC：14/14正确且eager达80%，最差{ratio:.3f}；24公开+18广播+90边界+27数值+30尾块+12UB+9特殊值+18通道压力+65536bool对+6bool广播+26共享Maximum通过，baseline待补',evidence=str(r/'minimum-REPORT.json'))
s['updated_at']=report['completed_at'];tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
print(json.dumps(report,ensure_ascii=False,indent=2))
