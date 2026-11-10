"""Serial guarded final SiluAndMul checks."""
import datetime,fcntl,hashlib,json,subprocess,sys
from pathlib import Path
r=Path(__file__).resolve().parent;root=r.parents[3]
snapshot=json.loads((r/'silu-and-mul-final-sources.json').read_text())['source_manifest']
def frozen():
 for p,h in snapshot.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
def receipt(label):
 g=json.loads((r/'requests'/label/'guard.json').read_text());assert g['exit_code']==0 and g['performance_valid'],label
def record(message):
 with (root/'.agent_memory/PROJECTS/tileops-ascend950/STATE.md').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);f.write('\n'+datetime.datetime.now(datetime.timezone.utc).isoformat()+' MAIN SiluAndMul '+message+'\n')
 print(message,flush=True)
receipt('silu-and-mul-extended-f16-v1')
jobs=[('silu-and-mul-extended-bf16-v1','silu_and_mul_extended.py',['--dtype','bfloat16']),('silu-and-mul-extended-f32-v1','silu_and_mul_extended.py',['--dtype','float32']),('silu-and-mul-manifest-v2','measure_silu_and_mul.py',['--output',str(r/'silu-and-mul-manifest-v2.jsonl')]),('silu-and-mul-public-v2','public_silu_and_mul.py',[])]
for label,script,args in jobs:
 frozen();p=r/'PLAN.json';d=json.loads(p.read_text());d.update(phase=label.replace('-','_'),active_request=label,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat());p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');record('启动'+label+'，父session见PLAN。')
 subprocess.run([sys.executable,str(r/'run_local.py'),label,script,*args],cwd=root,check=True)
 frozen();receipt(label);record(label+'通过，guard/source有效。')
 if label=='silu-and-mul-manifest-v2':
  rows=[json.loads(l) for l in (r/'silu-and-mul-manifest-v2.jsonl').read_text().splitlines()]
  assert len(rows)==4 and min(v['eager']['latency_ms']/v['latency_ms'] for v in rows)>=.8, 'manifest performance below threshold'
subprocess.run([sys.executable,str(r/'audit_silu_and_mul.py')],cwd=root,check=True)
p=r/'PLAN.json';d=json.loads(p.read_text());assert 'SiluAndMulFwdOp' not in d['local32_completed'];d['local32_completed'].append('SiluAndMulFwdOp');d.update(current_op='GeluAndMulFwdOp',phase='gelu_and_mul_preparation',active_request=None,session_id=None,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),last_verified='SiluAndMul audited:4 manifest,8 public,36 extended');p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');record('最终审计通过，64/70本地验证，16baselinepair待补。下一GeluAndMul尚未启动。')
