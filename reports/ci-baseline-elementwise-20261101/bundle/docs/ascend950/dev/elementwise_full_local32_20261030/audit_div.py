"""Final-source Div correctness, reference semantics and limited-core performance audit."""
import datetime,hashlib,json
from pathlib import Path
import xml.etree.ElementTree as ET
r=Path(__file__).resolve().parent;root=r.parents[3];op='DivFwdOp'
labels=['div-public-v4','div-ub-regression-v2','div-broadcast-regression-v1','div-boundaries-v1','div-numerical-final-v1','div-tail-stress-v1','div-manifest-v1']
for label in labels:
 g=json.loads((r/'requests'/label/'guard.json').read_text());assert g['exit_code']==0 and g['performance_valid'],label
rows=[json.loads(s) for s in (r/'div-manifest-v1.jsonl').read_text().splitlines()]
pairs=[v for v in json.loads((r/'baseline-scope.json').read_text())['pairs'] if v['op']==op]
assert len(rows)==16 and {v['case_id'] for v in rows}=={v['case'] for v in pairs}
for v in rows:
 assert v['correct'] and v['input_unchanged'] and v['verification']['checked_outputs']
 assert len(v['latency_samples_ms'])==len(v['eager']['latency_samples_ms'])==15
 assert v['runtime_limit']=={'cube_core_num':32,'vector_core_num':64}
 for path,h in v['source_manifest'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
source=root/'TileOPs/src/tileops/kernels/ascend/elementwise/arithmetic_div.py';h=hashlib.sha256(source.read_bytes()).hexdigest()
ref=r/'div_reference.py';rh=hashlib.sha256(ref.read_bytes()).hexdigest()
assert ref.read_bytes()==(root/'TileOPs/tests/ops/_div_reference.py').read_bytes()
dtypes={'torch.float16','torch.bfloat16','torch.float32'};modes={None,'trunc','floor'}
shapes={((),()),((0,3),(1,3)),((65,),(65,)),((129,),()),((2,1,5),(1,3,1)),((1,5),(3,1)),((3,5),(1,5)),((16387,),(16387,)),((2,4,64),(4,1)),((2,3,65),(3,1))}
ub_shapes={((2,11008),(1,11008)),((2,16320),(1,16320)),((2,16384),(1,16384)),((1,4,8192),(4,1))}
for suffix,n in [('ub-regression',12),('broadcast-regression',54),('boundaries',90),('numerical-semantics',27),('tail-stress',36)]:
 d=json.loads((r/('div-'+suffix+'.json')).read_text());assert d['module_sha256']==h and d['reference_sha256']==rh
 assert d['runtime_limit']=={'cube_core_num':32,'vector_core_num':64}
 checks=d['checks'];assert len(checks)==n and all(v['passed'] for v in checks)
 assert {v['dtype'] for v in checks}==dtypes
 for dtype in dtypes:
  typed=[v for v in checks if v['dtype']==dtype]
  if suffix=='ub-regression':
   assert len(typed)==4 and {(tuple(v['a_shape']),tuple(v['b_shape'])) for v in typed}==ub_shapes
   assert all(v['input_unchanged'] for v in typed)
   continue
  assert {v['rounding_mode'] for v in typed}==modes
  for mode in modes:
   subset=[v for v in typed if v['rounding_mode']==mode]
   if suffix=='broadcast-regression':
    assert len(subset)==6 and {(v['period'],v['rows']) for v in subset}=={(p,n) for p in [960,1088,2112] for n in [5,6]}
    assert all(v['sentinel'] and v['input_unchanged'] for v in subset)
   elif suffix=='boundaries':
    assert len(subset)==10 and {(tuple(v['a_shape']),tuple(v['b_shape'])) for v in subset}==shapes
    assert all(v['changed_inputs'] and v['input_unchanged'] for v in subset)
   elif suffix=='numerical-semantics':
    assert len(subset)==3 and {v['pattern'] for v in subset}=={'rolled','reversed','square'}
    assert all(v['values']==65537 and v['input_unchanged'] for v in subset)
   else:
    tails=[v for v in subset if 'check' not in v];assert len(tails)==3 and {v['n'] for v in tails}=={32767,524289,540671}
    assert all(v['changed_inputs'] and v['input_unchanged'] for v in tails)
    assert len(subset)==4 and sum(v.get('check')=='sentinel_and_negative_control' for v in subset)==1
for path,expected in json.loads((r/'div-public-v4-sources.json').read_text())['source_manifest'].items():
 assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,path
suites=list(ET.parse(r/'div-public.xml').getroot().iter('testsuite'));assert sum(int(s.get('tests',0)) for s in suites)==32
assert all(int(s.get('failures',0))+int(s.get('errors',0))+int(s.get('skipped',0))==0 for s in suites)
ratio=min(v['eager']['latency_ms']/v['latency_ms'] for v in rows);assert ratio>=.8
report={'op':op,'completed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'manifest_correct':16,'same_host_eager_80_percent_passes':16,'minimum_eager_ratio':ratio,'public_tests':32,'ub_regressions':12,'broadcast_regressions':54,'boundary_checks':90,'numerical_checks':27,'tail_and_sentinel_checks':36,'runtime_limit':{'cube':32,'vector':64},'registered_baseline_pairs':len(pairs),'full_registered_baseline_acceptance':False,'source_manifest':rows[0]['source_manifest'],'reference_sha256':rh,'reference_note':'Original TileOPs CUDA Div semantics: round-to-storage before trunc; FP32 remainder/quotient and storage floor/bump for floor. Ascend torch references differ on trunc and special values; CPU low-width floor and extreme-ratio vector fmod also disagree with scalar results. Preserved control failures and probes; div_reference uses scalar libm remainder and original dtype-dependent fast-tier limits. Original exact/bitwise checks retained.'}
(r/'div-REPORT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
p=r.parent/'elementwise_serial_20261030/STATE.json';s=json.loads(p.read_text())
for v in s['operators']:
 if v['op']==op:v.update(local32_validated=True,status='baseline_pending',detail=f'本地32AIC：16/16正确且eager达80%，最差{ratio:.3f}；32公开+12大广播UB+54广播+90边界+27数值+36尾块通过，baseline待补',evidence=str(r/'div-REPORT.json'))
s['updated_at']=report['completed_at'];tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
print(json.dumps(report,ensure_ascii=False,indent=2))
