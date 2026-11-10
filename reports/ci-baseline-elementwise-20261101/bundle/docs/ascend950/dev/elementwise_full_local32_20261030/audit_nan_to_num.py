"""Audit local32 NanToNum; registered baselines remain pending."""
import datetime,hashlib,json
from pathlib import Path
import xml.etree.ElementTree as ET
r=Path(__file__).resolve().parent
for label in ['nan-to-num-public-v1','nan-to-num-parameters-f16-v1','nan-to-num-parameters-bf16-v1','nan-to-num-parameters-f32-v1','nan-to-num-manifest-v1','nan-to-num-extended-v1']:
 g=json.loads((r/'requests'/label/'guard.json').read_text());assert g['exit_code']==0 and g['performance_valid'],label
for path,h in json.loads((r/'nan-to-num-final-sources.json').read_text())['source_manifest'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
h=hashlib.sha256((r.parents[3]/'TileOPs/src/tileops/kernels/ascend/elementwise/nan_to_num.py').read_bytes()).hexdigest()
limit={'cube_core_num':32,'vector_core_num':64}
rows=[json.loads(l) for l in (r/'nan-to-num-manifest-v1.jsonl').read_text().splitlines()]
pairs=[v for v in json.loads((r/'baseline-scope.json').read_text())['pairs'] if v['op']=='NanToNumFwdOp']
assert len(rows)==5 and {v['case_id'] for v in rows}=={v['case'] for v in pairs}
for v in rows:
 assert v['verification']['max_abs_err']==0 and v['max_error']==0
 assert v['op']=='NanToNumFwdOp' and v['correct'] and v['input_unchanged'] and v['verification']['checked_outputs']
 assert len(v['latency_samples_ms'])==len(v['eager']['latency_samples_ms'])==15
 assert v['runtime_limit']==limit and v['source_hash']==h
 for path,digest in v['source_manifest'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
ratio=min(v['eager']['latency_ms']/v['latency_ms'] for v in rows);assert ratio>=.8,ratio
suites=list(ET.parse(r/'nan-to-num-public.xml').getroot().iter('testsuite'));assert sum(int(s.get('tests',0)) for s in suites)==6
assert all(sum(int(s.get(k,0)) for k in ['failures','errors','skipped'])==0 for s in suites)
d=json.loads((r/'nan-to-num-extended.json').read_text());assert d['module_sha256']==h and d['runtime_limit']==limit
shapes={(0,3),(),(1,),(7,),(63,),(64,),(65,),(129,),(3,5),(2,1,5),(16387,),(32767,),(524289,),(540671,),(65536,)}
assert len(d['rows'])==45 and all(v['correct'] and v['input_unchanged'] for v in d['rows'])
assert {(v['mode'],v['dtype'],tuple(v['shape'])) for v in d['rows']}=={(m,t,s) for m in ['default'] for t in ['torch.float16','torch.bfloat16','torch.float32'] for s in shapes}
import struct
def bits(v):return None if v is None else struct.pack('!d',v).hex()
parameters=[(0.,None,None),(-0.,0.,-0.),(.3,.7,-.3),(float('inf'),-float('inf'),float('nan')),(float('nan'),float('nan'),float('nan')),(1e40,1e40,-1e40),(1e-50,1e-50,-1e-50),(0.,float('inf'),-float('inf')),(1e6,1e6,-1e6),(1.,None,-1.),(-1.,1.,None),(-0.,None,None)]
for dtype in ['float16','bfloat16','float32']:
 d=json.loads((r/('nan-to-num-parameters-'+dtype+'.json')).read_text());assert d['module_sha256']==h and d['runtime_limit']==limit
 assert len(d['checks'])==24 and all(v['passed'] and v['zero_sign'] and v['finite_bits'] for v in d['checks'])
 assert {(v['dtype'],v['nan_bits'],v['pos_bits'],v['neg_bits'],v['n']) for v in d['checks']}=={('torch.'+dtype,bits(nan),bits(pos),bits(neg),n) for nan,pos,neg in parameters for n in [65,1048576]}

report=dict(op='NanToNumFwdOp',completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),manifest_correct=5,same_host_eager_80_percent_passes=5,minimum_eager_ratio=ratio,public_tests=6,extended_checks=45,parameter_checks=72,runtime_limit={'cube':32,'vector':64},registered_baseline_pairs=len(pairs),full_registered_baseline_acceptance=False,source_manifest=rows[0]['source_manifest'])
(r/'nan-to-num-REPORT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
p=r.parent/'elementwise_serial_20261030/STATE.json';s=json.loads(p.read_text())
for v in s['operators']:
 if v['op']=='NanToNumFwdOp':v.update(local32_validated=True,status='baseline_pending',detail=f'本地32AIC：5/5正确且eager达80%，最差{ratio:.3f}；6公开/45扩展/72参数通过，baseline待补',evidence=str(r/'nan-to-num-REPORT.json'))
s['updated_at']=report['completed_at'];tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
print(json.dumps(report,ensure_ascii=False,indent=2))
