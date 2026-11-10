"""Serial guarded final GeluTanhAndMul checks."""
import datetime,fcntl,hashlib,json,subprocess,sys
from pathlib import Path
r=Path(__file__).resolve().parent;root=r.parents[3]
snapshot=json.loads((r/'gelu-tanh-and-mul-final-sources.json').read_text())['source_manifest']
def frozen():
 for p,h in snapshot.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
def receipt(label):
 g=json.loads((r/'requests'/label/'guard.json').read_text());assert g['exit_code']==0 and g['performance_valid'],label
def record(message):
 with (root/'.agent_memory/PROJECTS/tileops-ascend950/STATE.md').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);f.write('\n'+datetime.datetime.now(datetime.timezone.utc).isoformat()+' MAIN GeluTanhAndMul '+message+'\n')
 print(message,flush=True)
receipt('gelu-tanh-and-mul-extended-bf16-v2')
jobs=[('gelu-tanh-and-mul-extended-f16-v2','gelu_tanh_and_mul_extended.py',['--dtype','float16']),('gelu-tanh-and-mul-extended-f32-v1','gelu_tanh_and_mul_extended.py',['--dtype','float32']),('gelu-tanh-and-mul-manifest-v1','measure_gelu_tanh_and_mul.py',['--output',str(r/'gelu-tanh-and-mul-manifest-v1.jsonl')]),('gelu-tanh-and-mul-public-v2','public_gelu_tanh_and_mul.py',[])]
for label,script,args in jobs:
 frozen();p=r/'PLAN.json';d=json.loads(p.read_text());d.update(phase=label.replace('-','_'),active_request=label,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat());p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');record('启动'+label+'，父session见PLAN。')
 subprocess.run([sys.executable,str(r/'run_local.py'),label,script,*args],cwd=root,check=True)
 frozen();receipt(label);record(label+'通过，guard/source有效。')
 if label=='gelu-tanh-and-mul-manifest-v1':
  rows=[json.loads(l) for l in (r/'gelu-tanh-and-mul-manifest-v1.jsonl').read_text().splitlines()]
  assert len(rows)==3 and min(v['eager']['latency_ms']/v['latency_ms'] for v in rows)>=.8, 'manifest performance below threshold'
subprocess.run([sys.executable,str(r/'audit_gelu_tanh_and_mul.py')],cwd=root,check=True)
p=r/'PLAN.json';d=json.loads(p.read_text());assert 'GeluTanhAndMulFwdOp' not in d['local32_completed'];d['local32_completed'].append('GeluTanhAndMulFwdOp');d.update(current_op='AlibiFwdOp',phase='alibi_preparation',active_request=None,session_id=None,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),last_verified='GeluTanhAndMul audited:3 manifest,4 public,36 extended');p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');record('最终审计通过，66/70本地验证，4baselinepair待补。下一Alibi尚未启动。')
