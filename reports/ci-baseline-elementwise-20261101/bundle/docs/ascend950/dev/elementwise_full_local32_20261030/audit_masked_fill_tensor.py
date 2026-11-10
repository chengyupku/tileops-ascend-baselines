"""Frozen-source masked fill tensor audit under the local32 runtime limit."""
import json,hashlib,datetime
from pathlib import Path
import xml.etree.ElementTree as ET
r=Path(__file__).resolve().parent;op='MaskedFillTensorFwdOp';prefix='masked-fill-tensor'
for suffix in ['public-v2','bits','boundaries','tail-stress','manifest-v1']:
 g=json.loads((r/'requests'/(prefix+'-'+suffix)/'guard.json').read_text());assert g['exit_code']==0 and g['performance_valid'],suffix
rows=[json.loads(l) for l in (r/(prefix+'-manifest-v1.jsonl')).read_text().splitlines()]
pairs=[v for v in json.loads((r/'baseline-scope.json').read_text())['pairs'] if v['op']==op]
assert len(rows)==5 and {v['case_id'] for v in rows}=={v['case'] for v in pairs}
for v in rows:
 assert v['correct'] and v['input_unchanged'] and v['verification']['checked_outputs']
 assert len(v['latency_samples_ms'])==len(v['eager']['latency_samples_ms'])==15
 assert v['runtime_limit']=={'cube_core_num':32,'vector_core_num':64}
 for path,h in v['source_manifest'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
module=r.parents[3]/'TileOPs/src/tileops/kernels/ascend/elementwise/masked_fill_tensor.py';h=hashlib.sha256(module.read_bytes()).hexdigest()
dtypes={'torch.'+s for s in ['bool','uint8','int8','int16','int32','int64','float16','bfloat16','float32']}
for suffix,n,key in [('bits',90,'checks'),('boundaries',99,'rows'),('tail-stress',27,'checks')]:
 d=json.loads((r/(prefix+'-'+suffix+'.json')).read_text());assert d['module_sha256']==h and d['runtime_limit']=={'cube_core_num':32,'vector_core_num':64}
 assert len(d[key])==n and all(v.get('passed',v.get('correct',False)) for v in d[key])
 assert {v['dtype'] for v in d[key]}==dtypes
 for dtype in dtypes:
  subset=[v for v in d[key] if v['dtype']==dtype]
  assert len(subset)==n//9
  if suffix=='bits':
   assert {v['mode'] for v in subset}=={'all_false','all_true','mixed_raw','broadcast_raw'}
   assert all(v['values']==(131074 if v['mode']=='broadcast_raw' else 65537) for v in subset)
  elif suffix=='boundaries':
   assert sum(v.get('check')=='sentinel_and_negative_control' for v in subset)==1
   assert all(v['input_unchanged'] and v['runtime_value_changed'] and v['mask_changed'] for v in subset if 'input_shape' in v)
  else:assert all(v['changed_inputs'] and v['input_unchanged'] for v in subset)
suites=list(ET.parse(r/(prefix+'-public.xml')).getroot().iter('testsuite'))
assert sum(int(s.get('tests',0)) for s in suites)==36
assert all(int(s.get('errors',0))+int(s.get('failures',0))+int(s.get('skipped',0))==0 for s in suites)
ratios=[v['eager']['latency_ms']/v['latency_ms'] for v in rows];assert min(ratios)>=.8
report={'op':op,'completed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'manifest_correct':5,'same_host_eager_80_percent_passes':5,'minimum_eager_ratio':min(ratios),'public_tests':36,'boundary_checks':99,'bit_exact_checks':90,'large_tail_checks':27,'runtime_limit':{'cube':32,'vector':64},'registered_baseline_pairs':len(pairs),'full_registered_baseline_acceptance':False,'source_manifest':rows[0]['source_manifest'],'reference_note':'Public uint8/int16 use CPU masked_fill because aclnnInplaceMaskedFillTensor excludes these dtypes; original exact contract preserved.'}
(r/(prefix+'-REPORT.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
p=r.parent/'elementwise_serial_20261030/STATE.json';s=json.loads(p.read_text())
for v in s['operators']:
 if v['op']==op:v.update(local32_validated=True,status='baseline_pending',detail=f"本地32AIC：5/5正确且eager达80%，最差{min(ratios):.3f}；36公开+99边界+90逐位+27大尾部通过，baseline待补",evidence=str(r/(prefix+'-REPORT.json')))
s['updated_at']=report['completed_at'];tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
print(json.dumps(report,ensure_ascii=False,indent=2))
