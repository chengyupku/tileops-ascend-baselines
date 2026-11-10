"""Audit final-source local32 ReLU evidence; historical baselines stay pending."""
import datetime,hashlib,json
from pathlib import Path
import xml.etree.ElementTree as ET
r=Path(__file__).resolve().parent
for label in ['relu-special-simd-v1','relu-manifest-v2','relu-numerical-v2','relu-public-v2','relu-boundaries-v1','relu-tail-v1']:
 g=json.loads((r/'requests'/label/'guard.json').read_text());assert g['exit_code']==0 and g['performance_valid'],label
for path,h in json.loads((r/'relu-final-sources.json').read_text())['source_manifest'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
h=hashlib.sha256((r.parents[3]/'TileOPs/src/tileops/kernels/ascend/elementwise/relu.py').read_bytes()).hexdigest()
limit={'cube_core_num':32,'vector_core_num':64}
rows=[json.loads(l) for l in (r/'relu-manifest-v2.jsonl').read_text().splitlines()]
pairs=[v for v in json.loads((r/'baseline-scope.json').read_text())['pairs'] if v['op']=='ReluFwdOp']
assert len(rows)==3 and {v['case_id'] for v in rows}=={v['case'] for v in pairs}
for v in rows:
 assert v['op']=='ReluFwdOp' and v['correct'] and v['input_unchanged'] and v['verification']['checked_outputs']
 assert len(v['latency_samples_ms'])==len(v['eager']['latency_samples_ms'])==15
 assert v['runtime_limit']==limit and v['source_hash']==h
 for path,digest in v['source_manifest'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
ratio=min(v['eager']['latency_ms']/v['latency_ms'] for v in rows);assert ratio>=.8
suites=list(ET.parse(r/'relu-public.xml').getroot().iter('testsuite'));assert sum(int(s.get('tests',0)) for s in suites)==7
assert all(sum(int(s.get(k,0)) for k in ['failures','errors','skipped'])==0 for s in suites)
floats={'torch.float16','torch.bfloat16','torch.float32'}
d=json.loads((r/'relu-special-simd-v1.json').read_text());assert d['module_sha256']==h and d['runtime_limit']==limit
assert {(v['dtype'],v['n']) for v in d['checks']}=={(t,n) for t in floats for n in [64,65536]}
assert len(d['checks'])==6 and all(v['mismatches']==v['zero_sign_mismatches']==0 and v['input_unchanged'] for v in d['checks'])
d=json.loads((r/'relu-numerical.json').read_text());assert d['module_sha256']==h and d['runtime_limit']==limit
assert len(d['checks'])==16 and all(v['passed'] for v in d['checks'])
valid=[v for v in d['checks'] if not v.get('rejected')];assert len(valid)==6 and {(v['dtype'],v['inplace']) for v in valid}=={(t,i) for t in floats for i in [False,True]}
assert all(v['values']==65536 and v['zero_sign'] for v in valid)
rejected=[v for v in d['checks'] if v.get('rejected')];assert {(v['dtype'],v['inplace']) for v in rejected}=={('torch.'+t,i) for t in ['int8','uint8','int16','int32','int64'] for i in [False,True]}
d=json.loads((r/'relu-boundaries.json').read_text());assert d['runtime_limit']==limit
for path,digest in d['sources'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
assert len(d['rows'])==36 and all(v['correct'] for v in d['rows'])
shapes={(0,3),(),(1,),(7,),(63,),(64,),(65,),(129,),(3,5),(2,1,5),(16387,)}
for t in floats:
 subset=[v for v in d['rows'] if v['dtype']==t];assert len(subset)==12
 assert {tuple(v['shape']) for v in subset if 'shape' in v}==shapes
 assert all(v['input_unchanged'] for v in subset if 'shape' in v)
 assert sum(v.get('check')=='sentinel_and_negative_control' for v in subset)==1
d=json.loads((r/'relu-tail-stress.json').read_text());assert d['module_sha256']==h and d['runtime_limit']==limit
assert len(d['checks'])==9 and all(v['passed'] and v['changed_input'] for v in d['checks'])
assert {(v['dtype'],v['n']) for v in d['checks']}=={(t,n) for t in floats for n in [32767,524289,540671]}
report=dict(op='ReluFwdOp',completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),manifest_correct=3,same_host_eager_80_percent_passes=3,minimum_eager_ratio=ratio,public_tests=7,numerical_checks=6,unsupported_dtype_checks=10,special_value_checks=6,boundary_and_sentinel_checks=36,tail_checks=9,runtime_limit={'cube':32,'vector':64},registered_baseline_pairs=len(pairs),full_registered_baseline_acceptance=False,source_manifest=rows[0]['source_manifest'])
(r/'relu-REPORT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
p=r.parent/'elementwise_serial_20261030/STATE.json';s=json.loads(p.read_text())
for v in s['operators']:
 if v['op']=='ReluFwdOp':v.update(local32_validated=True,status='baseline_pending',detail=f'本地32AIC：3/3正确且eager达80%，最差{ratio:.3f}；7公开/6数值/10拒绝/6特殊值/36边界保护区/9尾块通过，baseline待补',evidence=str(r/'relu-REPORT.json'))
s['updated_at']=report['completed_at'];tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
print(json.dumps(report,ensure_ascii=False,indent=2))
