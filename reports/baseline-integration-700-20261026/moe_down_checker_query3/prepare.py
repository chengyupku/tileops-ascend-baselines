import json,hashlib,math
from pathlib import Path
H=Path(__file__).resolve().parent; P=H.parent
sources={}
for name,rel,ops in [('masked','bf16_moe_masked_recovery1/runtime/ASCEND_PROCESS_LOG_PATH/debug/plog/plog-3083225_20261027000806505.log',['aclnnIsClose_319','aclnnAll_321','aclnnIsClose_327','aclnnAll_329','aclnnIsClose_339','aclnnAll_341']),('contiguous','bf16_moe_hadamard_native10/qwen3-30b-ep8-decode-bfloat16/ASCEND_PROCESS_LOG_PATH/debug/plog/plog-3068522_20261026233604881.log',['aclnnIsClose_104','aclnnAll_106'])]:
 p=P/rel; records=[{'line':i,'text':l} for i,l in enumerate(p.read_text().splitlines(),1) if any('OpName:['+op+']' in l for op in ops) and 'Entering function params' in l]
 sources[name]={'path':str(p),'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest(),'records':records}
(H/'TRACE_EVIDENCE.json').write_text(json.dumps(sources,indent=2))
rows=json.loads((P/'moe_fullchecker_scratch5/PER_KEY_BOUNDS.json').read_text())['keys']; scope={r['case_id'] for r in json.loads((H/'SCOPE.json').read_text())}; rows=[r for r in rows if r['case_id'] in scope]
plans=[]; seen={}; alls={}
for operand in ['B','A','metadata','checker']:
 for row in rows:
  spec=dict(row['probe_descriptors'][operand]); n=math.prod(spec['shape']); spec.update(strides=[math.prod(spec['shape'][i+1:]) for i in range(len(spec['shape']))],storage_shape=[n],offset=0,format='NCL' if len(spec['shape'])==3 else 'ND',format_code=47 if len(spec['shape'])==3 else 2)
  k=json.dumps(spec,sort_keys=True)
  if k in seen: seen[k]['uses'].append([row['case_id'],operand]);continue
  id=f'p{len(plans):02d}_{operand}'; ak=json.dumps([spec['shape'],spec['strides'],spec['storage_shape'],spec['format']]); all_ref=alls.get(ak); alls.setdefault(ak,id)
  r={'id':id,**spec,'uses':[[row['case_id'],operand]],'query_all':all_ref is None,'all_reference':all_ref,'payload_bytes':n*(2*{'bfloat16':2,'int32':4,'float32':4}[spec['dtype']]+1)+1,'evidence':'contiguous' if operand=='A' else 'masked','scalar':{'shape':[],'strides':[],'storage_shape':[1],'format':'ND','format_code':2,'dtype':'bool'}}
  assert r['payload_bytes']<=20*2**30;seen[k]=r;plans.append(r)
(H/'PLAN.json').write_text(json.dumps(plans,indent=2));print([(r['id'],r['shape'],r['payload_bytes'],r['query_all']) for r in plans])
pins=[P/'moe_fullchecker_scratch5/PER_KEY_BOUNDS.json',P/'moe_fullchecker_scratch5/SOURCE_FINDINGS.json',P/'moe_v5_workspace_probe5/PARENT_REVIEW.json',Path('TileOPs/workloads/moe.py'),Path('TileOPs/workloads/numerics.py'),Path('TileOPs/src/tileops/manifest/spec/moe.yaml')]
sdk=Path('/usr/local/Ascend/cann-9.2.0-beta.2');pins += [sdk/'include'/s for s in ['acl/acl_base_rt.h','acl/acl_rt.h','aclnn/acl_meta.h','aclnn/aclnn_base.h','aclnnop/aclnn_isclose.h','aclnnop/aclnn_all.h']]
(H/'SOURCE_PINS.json').write_text(json.dumps({str(p.resolve()):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in pins},indent=2))
