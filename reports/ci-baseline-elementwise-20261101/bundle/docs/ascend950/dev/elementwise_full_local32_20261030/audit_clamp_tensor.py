"""Audit all local32 ClampTensor evidence; full registered baseline is separate."""
import datetime,hashlib,json,xml.etree.ElementTree as ET
from pathlib import Path
r=Path(__file__).resolve().parent;root=r.parents[3]
labels=['clamp-tensor-ieee-v2','clamp-tensor-manifest-both-v1','clamp-tensor-manifest-min-v1','clamp-tensor-manifest-max-v1','clamp-tensor-public-v1']
labels += ['clamp-tensor-'+suite+'-'+d+'-v1' for suite in ['broadcast','extended'] for d in ['f16','bf16','f32']]
for label in labels:
 g=json.loads((r/'requests'/label/'guard.json').read_text());assert g['exit_code']==0 and g['performance_valid'],label
snapshot=json.loads((r/'clamp-tensor-final-sources.json').read_text())['source_manifest']
for path,h in snapshot.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
h=hashlib.sha256((root/'TileOPs/src/tileops/kernels/ascend/elementwise/clamp_tensor.py').read_bytes()).hexdigest();limit={'cube_core_num':32,'vector_core_num':64}
rows=[]
for form in ['both','min','max']:
 part=[json.loads(l) for l in (r/('clamp-tensor-manifest-'+form+'-v1.jsonl')).read_text().splitlines()];assert len(part)==5;rows+=part
pairs=[v for v in json.loads((r/'baseline-scope.json').read_text())['pairs'] if v['op']=='ClampTensorFwdOp']
assert len(rows)==15 and len(pairs)==53 and {v['case_id'] for v in rows}=={v['case'] for v in pairs}
for v in rows:
 assert v['op']=='ClampTensorFwdOp' and v['correct'] and v['input_unchanged'] and v['verification']['checked_outputs']
 # Tensor clamp's original workload contract is zero tolerance. The manifest
 # measurements must also have observed zero absolute error, not just pass.
 assert v['verification']['max_abs_err']==0 and v['max_error']==0
 assert len(v['latency_samples_ms'])==len(v['eager']['latency_samples_ms'])==15
 assert v['runtime_limit']==limit and v['source_hash']==h
 for path,digest in v['source_manifest'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
ratio=min(v['eager']['latency_ms']/v['latency_ms'] for v in rows);assert ratio>=.8,ratio
suites=list(ET.parse(r/'clamp-tensor-public.xml').getroot().iter('testsuite'));assert sum(int(s.get('tests',0)) for s in suites)==19
assert all(sum(int(s.get(k,0)) for k in ['failures','errors','skipped'])==0 for s in suites)
def artifact(name):
 d=json.loads((r/name).read_text());assert d['module_sha256']==h and d['runtime_limit']==limit;return d['rows']
ieee=artifact('clamp-tensor-ieee.json');assert len(ieee)==27 and all(v['correct'] and v['input_unchanged'] and v['zero_sign_rule'] for v in ieee)
dtypes=['float16','bfloat16','float32'];modes=['both','min','max']
assert {(v['dtype'],v['form'],v['mode']) for v in ieee}=={('torch.'+d,f,m) for d in dtypes for f in ['tail','large','broadcast'] for m in modes}
shapes={(0,3),(),(1,),(7,),(63,),(64,),(65,),(129,),(3,5),(2,1,5),(16387,),(32767,),(524289,),(540671,),(65536,)}
broadcasts={((3,1),(1,5),()),((),(3,1),(1,5)),((2,1,7),(1,5,1),(7,)),((0,3),(1,3),()),((2,3,4,5),(1,3,1,1),(5,)),((1,1),(65,1),(1,3))}
for dtype in dtypes:
 ext=artifact('clamp-tensor-extended-'+dtype+'.json');assert len(ext)==45 and all(v['correct'] and v['input_unchanged'] for v in ext)
 assert {(v['dtype'],tuple(v['shape']),v['mode']) for v in ext}=={('torch.'+dtype,s,m) for s in shapes for m in modes}
 bc=artifact('clamp-tensor-broadcast-'+dtype+'.json');assert len(bc)==18 and all(v['correct'] and v['input_unchanged'] for v in bc)
 assert {(v['dtype'],tuple(tuple(s) for s in v['shapes']),v['mode']) for v in bc}=={('torch.'+dtype,s,m) for s in broadcasts for m in modes}
report=dict(op='ClampTensorFwdOp',completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),manifest_correct=15,same_host_eager_80_percent_passes=15,minimum_eager_ratio=ratio,public_tests=19,extended_checks=135,broadcast_checks=54,ieee_checks=27,runtime_limit={'cube':32,'vector':64},registered_baseline_pairs=len(pairs),full_registered_baseline_acceptance=False,source_manifest=rows[0]['source_manifest'],zero_sign_policy='Stable tie-to-bound on all kernel paths; CPU scalar/vector loops differ. Original Exact(atol=0,rtol=0) comparison is unchanged.',cpu_zero_sign_differences=[{'dtype':v['dtype'],'form':v['form'],'mode':v['mode']} for v in ieee if not v['cpu_zero_sign']],zero_sign_evidence=['clamp-tensor-cpu-probe.json','clamp-tensor-cpu-probe-inclusive.json','clamp-tensor-broadcast-probe.json','clamp-tensor-zero-layout-probe.json'])
(r/'clamp-tensor-REPORT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
p=r.parent/'elementwise_serial_20261030/STATE.json';s=json.loads(p.read_text())
for v in s['operators']:
 if v['op']=='ClampTensorFwdOp':v.update(local32_validated=True,status='baseline_pending',detail=f'本地32AIC：15/15正确且eager达80%，最差{ratio:.3f}；19公开/135扩展/54广播/27IEEE通过，baseline待补',evidence=str(r/'clamp-tensor-REPORT.json'))
s['updated_at']=report['completed_at'];tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
print(json.dumps(report,ensure_ascii=False,indent=2))
