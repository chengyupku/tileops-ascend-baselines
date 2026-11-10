"""Serial guarded final Softplus checks."""
import datetime,fcntl,hashlib,json,subprocess,sys
from pathlib import Path
r=Path(__file__).resolve().parent;root=r.parents[3]
snapshot=json.loads((r/'softplus-final-sources.json').read_text())['source_manifest']
def frozen():
 for p,h in snapshot.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
def receipt(label):
 g=json.loads((r/'requests'/label/'guard.json').read_text());assert g['exit_code']==0 and g['performance_valid'],label
def record(message):
 with (root/'.agent_memory/PROJECTS/tileops-ascend950/STATE.md').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);f.write('\n'+datetime.datetime.now(datetime.timezone.utc).isoformat()+' MAIN Softplus '+message+'\n')
 print(message,flush=True)
for label in ['softplus-edges-v2']:receipt(label)
for label,script,args in [('softplus-manifest-v2','measure_softplus.py',['--output',str(r/'softplus-manifest-v2.jsonl')]),('softplus-parameters-f16-v4','softplus_parameters.py',['--dtype','float16']),('softplus-parameters-bf16-v1','softplus_parameters.py',['--dtype','bfloat16']),('softplus-parameters-f32-v1','softplus_parameters.py',['--dtype','float32']),('softplus-public-v2','public_softplus.py',[]),('softplus-extended-v1','softplus_extended.py',[])]:
 frozen();p=r/'PLAN.json';d=json.loads(p.read_text());d.update(phase=label.replace('-','_'),active_request=label,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat());p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');record('启动'+label+'，父session见PLAN。')
 subprocess.run([sys.executable,str(r/'run_local.py'),label,script,*args],cwd=root,check=True)
 frozen();receipt(label);record(label+'通过，guard/source有效。')
 if label=='softplus-manifest-v2':
  rows=[json.loads(l) for l in (r/'softplus-manifest-v2.jsonl').read_text().splitlines()]
  assert len(rows)==4 and min(v['eager']['latency_ms']/v['latency_ms'] for v in rows)>=.8, 'manifest performance below threshold'



subprocess.run([sys.executable,str(r/'audit_softplus.py')],cwd=root,check=True)
p=r/'PLAN.json';d=json.loads(p.read_text());assert 'SoftplusFwdOp' not in d['local32_completed'];d['local32_completed'].append('SoftplusFwdOp');d.update(current_op='ClampTensorFwdOp',phase='clamp_tensor_preparation',active_request=None,session_id=None,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),last_verified='Softplus audited:4 manifest,3 public,45 extended,78 parameters,24 scalar edges');p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');record('最终审计通过，60/70本地验证，8baselinepair待补。下一ClampTensor尚未启动。')
