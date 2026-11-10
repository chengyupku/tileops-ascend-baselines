"""Serial foreground FloorDivide verification with separate host reservations."""
import datetime,fcntl,hashlib,json,subprocess,sys
from pathlib import Path
r=Path(__file__).resolve().parent;root=r.parents[3]
snapshot=json.loads((r/'floor-divide-public-v2-sources.json').read_text())['source_manifest']
def frozen():
 for path,h in snapshot.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
def receipt(label):
 g=json.loads((r/'requests'/label/'guard.json').read_text());assert g['exit_code']==0 and g['performance_valid'],label
 assert 'race remains' not in (r/'requests'/label/'output.log').read_text(),label

def record(message):
 with (root/'.agent_memory/PROJECTS/tileops-ascend950/STATE.md').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);f.write('\n'+datetime.datetime.now(datetime.timezone.utc).isoformat()+' MAIN FloorDivide '+message+'\n')
 print(message,flush=True)
jobs=[('floor-divide-public-v2','public_floor_divide.py',None,20),
 ('floor-divide-ub-regression-v2','floor_divide_ub_regression.py','ub-regression',12),
 ('floor-divide-broadcast-regression-v1','floor_divide_broadcast_regression.py','broadcast-regression',18),
 ('floor-divide-boundaries-v1','floor_divide_boundaries.py','boundaries',30),
 ('floor-divide-numerical-v2','floor_divide_numerical.py','numerical',9),
 ('floor-divide-tail-stress-v1','floor_divide_tail_stress.py','tail-stress',12),
 ('floor-divide-fast-boundaries-v2','floor_divide_fast_boundaries.py','fast-boundaries',27),
 ('floor-divide-repair-v1','floor_divide_repair.py','repair',12),
 ('floor-divide-shared-div-public-v1','public_div.py',None,32),
 ('floor-divide-shared-remainder-public-v1','public_remainder.py',None,20),
 ('floor-divide-manifest-v2','measure_floor_divide.py',None,14)]
for label,script,suffix,count in jobs:
 frozen();p=r/'PLAN.json';d=json.loads(p.read_text());d.update(phase=label.replace('-','_'),active_request=label,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat());p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
 record('启动 '+label+'，串行父session见PLAN.session_id。')
 args=[sys.executable,str(r/'run_local.py'),label,script]
 if script=='measure_floor_divide.py':args+=['--output',str(r/'floor_divide-manifest-v2.jsonl')]
 subprocess.run(args,cwd=root,check=True);frozen();receipt(label)
 if suffix:
  result=json.loads((r/f'floor_divide-{suffix}.json').read_text());assert len(result['checks'])==count and all(v['passed'] for v in result['checks'])
  assert result['runtime_limit']=={'cube_core_num':32,'vector_core_num':64}
  assert result['module_sha256']==hashlib.sha256((root/'TileOPs/src/tileops/kernels/ascend/elementwise/arithmetic_modulo.py').read_bytes()).hexdigest()
  assert result['reference_sha256']==hashlib.sha256((r/'floor_divide_reference.py').read_bytes()).hexdigest()
  for path,h in result['dependency_manifest'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
 elif script.startswith('public_'):
  import xml.etree.ElementTree as ET
  name={'public_floor_divide.py':'floor_divide-public.xml','public_div.py':'div-public.xml','public_remainder.py':'remainder-public.xml'}[script]
  suites=list(ET.parse(r/name).getroot().iter('testsuite'));assert sum(int(s.get('tests',0)) for s in suites)==count
  assert all(int(s.get('failures',0))+int(s.get('errors',0))+int(s.get('skipped',0))==0 for s in suites)
 record(label+' 完成，guard有效/冻结源码一致/无同步race警告。')
frozen();subprocess.run([sys.executable,str(r/'audit_floor_divide.py')],cwd=root,check=True)
p=r/'PLAN.json';d=json.loads(p.read_text());assert 'FloorDivideFwdOp' not in d['local32_completed'];d['local32_completed'].append('FloorDivideFwdOp');d.update(current_op='LerpScalarFwdOp',phase='lerp_scalar_preparation',active_request=None,session_id=None,last_verified='FloorDivide audited:14 manifest,20 public,12 UB,18 broadcast,30 boundaries,9 numerical,12 tail,27 fast boundaries,12 repair,32 Div+20 Remainder shared regression',updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat());p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
record('最终审计通过，PLAN/ledger40/70本地验证，14登记baselinepair待补。报告floor_divide-REPORT.json。下一项LerpScalar准备，尚未启动；全目标active。')
