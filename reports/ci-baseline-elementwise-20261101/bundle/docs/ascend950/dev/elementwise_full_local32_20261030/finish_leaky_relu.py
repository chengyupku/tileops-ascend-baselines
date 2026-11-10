"""Serial guarded final LeakyRelu checks."""
import datetime,fcntl,hashlib,json,subprocess,sys
from pathlib import Path
r=Path(__file__).resolve().parent;root=r.parents[3]
snapshot=json.loads((r/'leaky_relu-final-sources.json').read_text())['source_manifest']
def frozen():
 for p,h in snapshot.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
def receipt(label):
 g=json.loads((r/'requests'/label/'guard.json').read_text());assert g['exit_code']==0 and g['performance_valid'],label
def record(message):
 with (root/'.agent_memory/PROJECTS/tileops-ascend950/STATE.md').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);f.write('\n'+datetime.datetime.now(datetime.timezone.utc).isoformat()+' MAIN LeakyRelu '+message+'\n')
 print(message,flush=True)
for label in ['leaky-relu-manifest-v2']:receipt(label)
for label,script,args in [('leaky-relu-public-v2','public_leaky_relu.py',[]),('leaky-relu-parameters-f16-v2','leaky_relu_parameters.py',['--dtype','float16']),('leaky-relu-parameters-bf16-v2','leaky_relu_parameters.py',['--dtype','bfloat16']),('leaky-relu-parameters-f32-v2','leaky_relu_parameters.py',['--dtype','float32']),('leaky-relu-extended-v1','leaky_relu_extended.py',[])]:
 frozen();p=r/'PLAN.json';d=json.loads(p.read_text());d.update(phase=label.replace('-','_'),active_request=label,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat());p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');record('启动'+label+'，父session见PLAN。')
 subprocess.run([sys.executable,str(r/'run_local.py'),label,script,*args],cwd=root,check=True)
 frozen();receipt(label);record(label+'通过，guard/source有效。')

subprocess.run([sys.executable,str(r/'audit_leaky_relu.py')],cwd=root,check=True)
p=r/'PLAN.json';d=json.loads(p.read_text());assert 'LeakyReluFwdOp' not in d['local32_completed'];d['local32_completed'].append('LeakyReluFwdOp');d.update(current_op='EluFwdOp',phase='elu_preparation',active_request=None,session_id=None,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),last_verified='LeakyRelu audited:4 manifest,7 public,45 extended,54 parameters');p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');record('最终审计通过，57/70本地验证，8baselinepair待补。下一Elu尚未启动。')
