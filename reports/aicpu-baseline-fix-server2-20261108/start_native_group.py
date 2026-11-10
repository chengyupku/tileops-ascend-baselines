"""CPU dependency gate then one bounded foreground guard for eight formal cases."""
import pathlib,json,time,subprocess,sys,os
P=pathlib.Path(__file__).resolve().parent;deadline=time.monotonic()+900
prior=P/'native-v2/runs/2fea74995c281f088e0e/a1/guard.json'
while True:
 try:g=json.load(open(prior))
 except (FileNotFoundError,json.JSONDecodeError):g={}
 if g.get('phase') in ('finished','released_without_launch'):
  assert g.get('workload_exit_code')==0 and g.get('performance_valid'),g
  break
 if time.monotonic()>deadline:raise RuntimeError('second case not complete; group not started')
 time.sleep(1)
os.environ['LD_LIBRARY_PATH']='/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64'
receipt=P/'native-group-guard.json';assert not receipt.exists()
cmd=['python3','/opt/torch-v010-20261106/tools/npu_guard.py','run','--owner','root/aicpu-fix-server2/native-group8','--wait','600','--timeout','900','--receipt',str(receipt),'--',sys.executable,str(P/'native_group.py')]
r=subprocess.run(cmd)
if r.returncode:raise SystemExit(r.returncode)
g=json.load(open(receipt));assert g['performance_valid'] and g['workload_exit_code']==0
rows=json.load(open(P/'NATIVE_GROUP_MEMBERS.json'));assert len(rows)==8 and all(v['exit_code']==0 for v in rows)
for row in rows:
 out=pathlib.Path(row['out']);assert not (out/'guard.json').exists();(out/'guard.json').write_bytes(receipt.read_bytes());(out/'group_member.json').write_text(json.dumps(row,indent=2))
print('GROUP COMPLETE8',flush=True)
