"""Serial guarded final Mish checks."""
import datetime,fcntl,hashlib,json,subprocess,sys
from pathlib import Path
r=Path(__file__).resolve().parent;root=r.parents[3]
snapshot=json.loads((r/'mish-final-sources.json').read_text())['source_manifest']
def frozen():
 for p,h in snapshot.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
def receipt(label):
 g=json.loads((r/'requests'/label/'guard.json').read_text());assert g['exit_code']==0 and g['performance_valid'],label
def record(message):
 with (root/'.agent_memory/PROJECTS/tileops-ascend950/STATE.md').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);f.write('\n'+datetime.datetime.now(datetime.timezone.utc).isoformat()+' MAIN Mish '+message+'\n')
 print(message,flush=True)
for label in ['mish-ieee-v1']:receipt(label)
for label,script,args in [('mish-manifest-v1','measure_mish.py',['--output',str(r/'mish-manifest-v1.jsonl')]),('mish-public-v2','public_mish.py',[]),('mish-extended-v1','mish_extended.py',[])]:
 frozen();p=r/'PLAN.json';d=json.loads(p.read_text());d.update(phase=label.replace('-','_'),active_request=label,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat());p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');record('启动'+label+'，父session见PLAN。')
 subprocess.run([sys.executable,str(r/'run_local.py'),label,script,*args],cwd=root,check=True)
 frozen();receipt(label);record(label+'通过，guard/source有效。')
 if label=='mish-manifest-v1':
  rows=[json.loads(l) for l in (r/'mish-manifest-v1.jsonl').read_text().splitlines()]
  assert len(rows)==4 and min(v['eager']['latency_ms']/v['latency_ms'] for v in rows)>=.8, 'manifest performance below threshold'

subprocess.run([sys.executable,str(r/'audit_mish.py')],cwd=root,check=True)
p=r/'PLAN.json';d=json.loads(p.read_text());assert 'MishFwdOp' not in d['local32_completed'];d['local32_completed'].append('MishFwdOp');d.update(current_op='SeluFwdOp',phase='selu_preparation',active_request=None,session_id=None,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),last_verified='Mish audited:4 manifest,5 public,45 extended,9 IEEE');p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');record('最终审计通过，55/70本地验证，8baselinepair待补。下一Selu尚未启动。')
