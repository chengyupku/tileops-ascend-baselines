"""Audit local32 LerpTensor; registered baseline acceptance remains pending."""
import datetime,hashlib,json
from pathlib import Path
import xml.etree.ElementTree as ET
r=Path(__file__).resolve().parent
for label in ['lerp-tensor-public-v1','lerp-tensor-manifest-v1','lerp-tensor-extended-f16-v1','lerp-tensor-extended-bf16-v1','lerp-tensor-extended-f32-v1']:
 g=json.loads((r/'requests'/label/'guard.json').read_text());assert g['exit_code']==0 and g['performance_valid'],label
for path,h in json.loads((r/'lerp-tensor-final-sources.json').read_text())['source_manifest'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
h=hashlib.sha256((r.parents[3]/'TileOPs/src/tileops/kernels/ascend/elementwise/lerp.py').read_bytes()).hexdigest()
limit={'cube_core_num':32,'vector_core_num':64}
rows=[json.loads(l) for l in (r/'lerp-tensor-manifest-v1.jsonl').read_text().splitlines()]
pairs=[v for v in json.loads((r/'baseline-scope.json').read_text())['pairs'] if v['op']=='LerpTensorFwdOp']
assert len(rows)==5 and len({v['case_id'] for v in rows})==5 and {v['case_id'] for v in rows}=={v['case'] for v in pairs}
for v in rows:
 assert v['op']=='LerpTensorFwdOp' and v['correct'] and v['input_unchanged'] and v['verification']['checked_outputs']
 assert len(v['latency_samples_ms'])==len(v['eager']['latency_samples_ms'])==15
 assert v['runtime_limit']==limit and v['source_hash']==h
 for path,digest in v['source_manifest'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
ratio=min(v['eager']['latency_ms']/v['latency_ms'] for v in rows);assert ratio>=.8,ratio
suites=list(ET.parse(r/'lerp-tensor-public-v1.xml').getroot().iter('testsuite'));count=sum(int(s.get('tests',0)) for s in suites);assert count==9,count
assert all(sum(int(s.get(k,0)) for k in ['failures','errors','skipped'])==0 for s in suites)
shapes=[(0,3),(),(1,),(7,),(63,),(64,),(65,),(129,),(3,5),(2,1,5),(16387,),(32767,),(524289,),(540671,),(65536,)]
broadcast=[((1,7),(5,1),()),((),(3,5),(1,5)),((2,1,5),(1,3,1),(2,3,5)),((3,1),(1,5),(3,5)),((0,1),(1,5),()),((1,),(129,),(1,)),((17,1),(1,769),())]
for dtype in ['float16','bfloat16','float32']:
 d=json.loads((r/('lerp-tensor-extended-'+dtype+'.json')).read_text());assert d['module_sha256']==h and d['runtime_limit']==limit
 assert len(d['rows'])==22 and all(v['correct'] and v['input_unchanged'] and v['dtype']=='torch.'+dtype for v in d['rows'])
 assert {(tuple(v['input_shape']),tuple(v['end_shape']),tuple(v['weight_shape'])) for v in d['rows']}==set([(s,s,s) for s in shapes]+broadcast)
report=dict(op='LerpTensorFwdOp',completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),manifest_correct=5,same_host_eager_80_percent_passes=5,minimum_eager_ratio=ratio,public_tests=count,extended_checks=66,runtime_limit={'cube':32,'vector':64},registered_baseline_pairs=len(pairs),full_registered_baseline_acceptance=False,source_manifest=rows[0]['source_manifest'])
(r/'lerp-tensor-REPORT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
p=r.parent/'elementwise_serial_20261030/STATE.json';s=json.loads(p.read_text())
for v in s['operators']:
 if v['op']=='LerpTensorFwdOp':v.update(local32_validated=True,status='baseline_pending',detail=f'本地32AIC：5/5正确且eager达80%，最差{ratio:.3f}；{count}公开/66扩展通过，baseline待补',evidence=str(r/'lerp-tensor-REPORT.json'))
s['updated_at']=report['completed_at'];tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
print(json.dumps(report,ensure_ascii=False,indent=2))
