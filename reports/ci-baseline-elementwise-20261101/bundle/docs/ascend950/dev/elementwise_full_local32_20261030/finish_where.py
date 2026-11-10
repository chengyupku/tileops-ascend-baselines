"""Serial guarded final Where checks."""
import datetime,fcntl,hashlib,json,subprocess,sys
from pathlib import Path
r=Path(__file__).resolve().parent;root=r.parents[3]
snapshot=json.loads((r/'where-final-sources.json').read_text())['source_manifest']
def frozen():
 for p,h in snapshot.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
def receipt(label):
 g=json.loads((r/'requests'/label/'guard.json').read_text());assert g['exit_code']==0 and g['performance_valid'],label
def record(message):
 with (root/'.agent_memory/PROJECTS/tileops-ascend950/STATE.md').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);f.write('\n'+datetime.datetime.now(datetime.timezone.utc).isoformat()+' MAIN Where '+message+'\n')
 print(message,flush=True)
receipt('where-public-v3')
jobs=[('where-manifest-v3','measure_where.py',['--output',str(r/'where-manifest-v3.jsonl')]),('where-extended-f16-v3','where_extended.py',['--dtype','float16']),('where-extended-bf16-v2','where_extended.py',['--dtype','bfloat16']),('where-extended-f32-v2','where_extended.py',['--dtype','float32'])]
for label,script,args in jobs:
 frozen();p=r/'PLAN.json';d=json.loads(p.read_text());d.update(phase=label.replace('-','_'),active_request=label,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat());p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');record('启动'+label+'，父session见PLAN。')
 subprocess.run([sys.executable,str(r/'run_local.py'),label,script,*args],cwd=root,check=True)
 frozen();receipt(label);record(label+'通过，guard/source有效。')
 if label=='where-manifest-v3':
  rows=[json.loads(l) for l in (r/'where-manifest-v3.jsonl').read_text().splitlines()]
  assert len(rows)==5 and min(v['eager']['latency_ms']/v['latency_ms'] for v in rows)>=.8, 'manifest performance below threshold'
subprocess.run([sys.executable,str(r/'audit_where.py')],cwd=root,check=True)
p=r/'PLAN.json';d=json.loads(p.read_text());assert 'WhereFwdOp' not in d['local32_completed'];d['local32_completed'].append('WhereFwdOp');d.update(current_op='LerpTensorFwdOp',phase='lerp_tensor_preparation',active_request=None,session_id=None,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),last_verified='Where audited:5 manifest,16 public,66 extended');p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');record('最终审计通过，69/70本地验证，20baselinepair待补。下一LerpTensor尚未启动。')
