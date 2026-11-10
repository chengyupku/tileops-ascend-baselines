"""Foreground serial final verification; each device command has its own host guard."""
import datetime,fcntl,hashlib,json,subprocess,sys
from pathlib import Path
r=Path(__file__).resolve().parent
root=r.parents[3]
module=root/'TileOPs/src/tileops/kernels/ascend/elementwise/arithmetic_power.py'
ref=r/'pow_reference.py'
h=hashlib.sha256(module.read_bytes()).hexdigest()
rh=hashlib.sha256(ref.read_bytes()).hexdigest()
def frozen():
 assert hashlib.sha256(module.read_bytes()).hexdigest()==h
 assert hashlib.sha256(ref.read_bytes()).hexdigest()==rh
def receipt(label):
 g=json.loads((r/'requests'/label/'guard.json').read_text())
 assert g['exit_code']==0 and g['performance_valid'],label
def record(message):
 with (root/'.agent_memory/PROJECTS/tileops-ascend950/STATE.md').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX)
  f.write('\n'+datetime.datetime.now(datetime.timezone.utc).isoformat()+' MAIN Pow '+message+'\n')
 print(message,flush=True)
receipt('pow-public-v1')
for path,expected in json.loads((r/'pow-public-v1-sources.json').read_text())['source_manifest'].items():
 assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,path
jobs=[('pow-ub-regression-v2','pow_ub_regression.py','ub-regression',12),
 ('pow-broadcast-regression-v1','pow_broadcast_regression.py','broadcast-regression',18),
 ('pow-boundaries-v1','pow_boundaries.py','boundaries',30),
 ('pow-numerical-v1','pow_numerical.py','numerical',27),
 ('pow-tail-stress-v1','pow_tail_stress.py','tail-stress',12),
 ('pow-manifest-v2','measure_pow.py',None,14)]
for label,script,suffix,count in jobs:
 frozen()
 p=r/'PLAN.json';d=json.loads(p.read_text());d.update(phase=label.replace('-','_'),active_request=label,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat());p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
 record('启动 '+label+'；finish_pow.py串行前台，主会话session见PLAN.session_id。')
 args=[sys.executable,str(r/'run_local.py'),label,script]
 if suffix is None:args+=['--output',str(r/'pow-manifest-v2.jsonl')]
 subprocess.run(args,cwd=root,check=True)
 frozen();receipt(label)
 if suffix is not None:
  result=json.loads((r/f'pow-{suffix}.json').read_text());assert result['module_sha256']==h and result['reference_sha256']==rh
  assert result['runtime_limit']=={'cube_core_num':32,'vector_core_num':64}
  assert len(result['checks'])==count and all(v['passed'] for v in result['checks'])
 record(label+' 完成，guard有效及冻结源码一致。')
frozen()
subprocess.run([sys.executable,str(r/'audit_pow.py')],cwd=root,check=True)
p=r/'PLAN.json';d=json.loads(p.read_text());assert 'PowFwdOp' not in d['local32_completed'];d['local32_completed'].append('PowFwdOp');d.update(current_op='FloorDivideFwdOp',phase='floor_divide_preparation',active_request=None,session_id=None,last_verified='Pow audited:14 manifest,9 public,27 numerical,12 UB,18 broadcast,30 boundaries,12 tail,9 special,18 channel',updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat());p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
record('audit_pow.py通过，PLAN/ledger39/70本地验证；42登记baselinepair待补。报告pow-REPORT.json。下一项FloorDivideFwdOp，尚未启动。全目标active。')
