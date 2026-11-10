"""Audit complete local32 Alibi checks, preserving baseline-pending status."""
import datetime,hashlib,json
from pathlib import Path
import xml.etree.ElementTree as ET
r=Path(__file__).resolve().parent
labels=['alibi-extended-f16-v1','alibi-extended-bf16-v1','alibi-extended-f32-v3','alibi-manifest-v1','alibi-public-v3']
for label in labels:
 g=json.loads((r/'requests'/label/'guard.json').read_text());assert g['exit_code']==0 and g['performance_valid'],label
for p,h in json.loads((r/'alibi-final-sources.json').read_text())['source_manifest'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
h=hashlib.sha256((r.parents[3]/'TileOPs/src/tileops/kernels/ascend/elementwise/alibi.py').read_bytes()).hexdigest();limit={'cube_core_num':32,'vector_core_num':64}
rows=[json.loads(l) for l in (r/'alibi-manifest-v1.jsonl').read_text().splitlines()]
pairs=[v for v in json.loads((r/'baseline-scope.json').read_text())['pairs'] if v['op']=='AlibiFwdOp']
assert len(rows)==4 and len({v['case_id'] for v in rows})==4
assert {(v['parameters']['seq_len'],v['parameters']['num_heads'],v['parameters']['out_dtype']) for v in rows}=={(seq,32,'torch.'+dt) for seq in [2048,4096] for dt in ['float16','bfloat16']}
for v in rows:
 assert v['op']=='AlibiFwdOp' and v['correct'] and v['input_unchanged'] and v['verification']['checked_outputs']
 assert len(v['latency_samples_ms'])==len(v['eager']['latency_samples_ms'])==15
 assert v['runtime_limit']==limit and v['source_hash']==h
 for p,digest in v['source_manifest'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest,p
ratio=min(v['eager']['latency_ms']/v['latency_ms'] for v in rows);assert ratio>=.8,ratio
suites=list(ET.parse(r/'alibi-public-v3.xml').getroot().iter('testsuite'));assert sum(int(s.get('tests',0)) for s in suites)==2
assert all(sum(int(s.get(k,0)) for k in ['failures','errors','skipped'])==0 for s in suites)
shapes={(1,1),(1,3),(3,5),(7,3),(31,7),(63,3),(64,8),(65,9),(129,3),(257,5),(513,2),(17,33)}
for dtype in ['float16','bfloat16','float32']:
 d=json.loads((r/('alibi-extended-'+dtype+'.json')).read_text());assert d['module_sha256']==h and d['runtime_limit']==limit
 assert len(d['rows'])==12 and all(v['correct'] and v['fresh_output'] and v['symmetric'] and v['negative_zero_diagonal'] for v in d['rows'])
 assert {(v['dtype'],v['seq_len'],v['num_heads']) for v in d['rows']}=={('torch.'+dtype,seq,heads) for seq,heads in shapes}
 assert {tuple(v) for v in d['invalid_rejected']}=={(0,3),(-1,3),(3,0),(3,-1)}

report=dict(op='AlibiFwdOp',completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),manifest_correct=4,same_host_eager_80_percent_passes=4,minimum_eager_ratio=ratio,public_tests=2,extended_checks=36,runtime_limit={'cube':32,'vector':64},registered_baseline_pairs=len(pairs),full_registered_baseline_acceptance=False,source_manifest=rows[0]['source_manifest'])
(r/'alibi-REPORT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
p=r.parent/'elementwise_serial_20261030/STATE.json';s=json.loads(p.read_text())
for v in s['operators']:
 if v['op']=='AlibiFwdOp':v.update(local32_validated=True,status='baseline_pending',detail=f'本地32AIC：4/4正确且eager达80%，最差{ratio:.3f}；2公开/36扩展通过，baseline待补',evidence=str(r/'alibi-REPORT.json'))
s['updated_at']=report['completed_at'];tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
print(json.dumps(report,ensure_ascii=False,indent=2))
