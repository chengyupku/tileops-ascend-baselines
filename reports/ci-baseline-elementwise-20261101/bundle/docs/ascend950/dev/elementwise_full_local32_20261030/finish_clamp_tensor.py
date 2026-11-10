"""Serial guarded final ClampTensor checks."""
import datetime,fcntl,hashlib,json,subprocess,sys
from pathlib import Path
r=Path(__file__).resolve().parent;root=r.parents[3]
snapshot=json.loads((r/'clamp-tensor-final-sources.json').read_text())['source_manifest']
def frozen():
 for p,h in snapshot.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
def receipt(label):
 g=json.loads((r/'requests'/label/'guard.json').read_text());assert g['exit_code']==0 and g['performance_valid'],label
def record(message):
 with (root/'.agent_memory/PROJECTS/tileops-ascend950/STATE.md').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);f.write('\n'+datetime.datetime.now(datetime.timezone.utc).isoformat()+' MAIN ClampTensor '+message+'\n')
 print(message,flush=True)
def performance(form):
 rows=[json.loads(l) for l in (r/('clamp-tensor-manifest-'+form+'-v1.jsonl')).read_text().splitlines()]
 assert len(rows)==5 and min(v['eager']['latency_ms']/v['latency_ms'] for v in rows)>=.8,'manifest performance below threshold'
for label in ['clamp-tensor-ieee-v2','clamp-tensor-manifest-both-v1']:receipt(label)
performance('both')
jobs=[('clamp-tensor-manifest-'+f+'-v1','measure_clamp_tensor.py',['--regex=-'+f+'-only-','--output',str(r/('clamp-tensor-manifest-'+f+'-v1.jsonl'))]) for f in ['min','max']]
jobs += [('clamp-tensor-public-v1','public_clamp_tensor.py',[])]
jobs += [('clamp-tensor-'+suite+'-'+short+'-v1','clamp_tensor_'+suite+'.py',['--dtype',dtype]) for suite in ['broadcast','extended'] for short,dtype in [('f16','float16'),('bf16','bfloat16'),('f32','float32')]]
for label,script,args in jobs:
 frozen();p=r/'PLAN.json';d=json.loads(p.read_text());d.update(phase=label.replace('-','_'),active_request=label,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat());p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');record('启动'+label+'，父session见PLAN。')
 subprocess.run([sys.executable,str(r/'run_local.py'),label,script,*args],cwd=root,check=True)
 frozen();receipt(label);record(label+'通过，guard/source有效。')
 if label.startswith('clamp-tensor-manifest-'):performance(label.split('-')[-2])
subprocess.run([sys.executable,str(r/'audit_clamp_tensor.py')],cwd=root,check=True)
p=r/'PLAN.json';d=json.loads(p.read_text());assert 'ClampTensorFwdOp' not in d['local32_completed'];d['local32_completed'].append('ClampTensorFwdOp');d.update(current_op='ClampScalarFwdOp',phase='clamp_scalar_preparation',active_request=None,session_id=None,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),last_verified='ClampTensor audited:15 manifest,19 public,135 extended,54 broadcast,27 IEEE');p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');record('最终审计通过，61/70本地验证，53baselinepair待补。下一ClampScalar尚未启动。')
