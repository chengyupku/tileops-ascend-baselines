"""Serial guarded final ReLU checks."""
import datetime,fcntl,hashlib,json,subprocess,sys
from pathlib import Path
r=Path(__file__).resolve().parent;root=r.parents[3]
snapshot=json.loads((r/'relu-final-sources.json').read_text())['source_manifest']
def frozen():
 for p,h in snapshot.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
def receipt(label):
 g=json.loads((r/'requests'/label/'guard.json').read_text());assert g['exit_code']==0 and g['performance_valid'],label
def record(message):
 with (root/'.agent_memory/PROJECTS/tileops-ascend950/STATE.md').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);f.write('\n'+datetime.datetime.now(datetime.timezone.utc).isoformat()+' MAIN Relu '+message+'\n')
 print(message,flush=True)
for label in ['relu-special-simd-v1','relu-manifest-v2','relu-numerical-v2','relu-public-v2']:receipt(label)
for label,script in [('relu-boundaries-v1','relu_boundaries.py'),('relu-tail-v1','relu_tail_stress.py')]:
 frozen();p=r/'PLAN.json';d=json.loads(p.read_text());d.update(phase=label.replace('-','_'),active_request=label,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat());p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');record('启动'+label+'，父session见PLAN。')
 subprocess.run([sys.executable,str(r/'run_local.py'),label,script],cwd=root,check=True)
 frozen();receipt(label);record(label+'通过，guard/source有效。')
subprocess.run([sys.executable,str(r/'audit_relu.py')],cwd=root,check=True)
p=r/'PLAN.json';d=json.loads(p.read_text());assert 'ReluFwdOp' not in d['local32_completed'];d['local32_completed'].append('ReluFwdOp');d.update(current_op='GeluFwdOp',phase='gelu_preparation',active_request=None,session_id=None,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),last_verified='Relu audited:3 manifest,7 public,6 numerical,10 rejection,6 IEEE,36 boundaries/sentinels,9 tails');p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');record('最终审计通过，50/70本地验证，15baselinepair待补。下一Gelu尚未启动。')
