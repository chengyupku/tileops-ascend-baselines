R=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_guarded_resume5
cat /run/lock/ascend-npu/device-0.lock
tail -8 "$R/compile1.log"
ps -eo pid,ppid,pgid,etime,pcpu,args | grep -E 'asc_opc|bisheng|flash_attn_ascend950_1|stage/bin/make' | head -15
/opt/tileops-ci/runtime/bin/python3 - <<'PY'
import importlib.metadata as m,json,pathlib
p='/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_native_build5/attempt01/stage/python'
roots=[p,'/opt/tileops-ci/runtime/lib/python3.12/site-packages']
rows=[]
for root in roots:
 for d in m.distributions(path=[root]):
  if d.metadata['Name'].lower().replace('_','-') in ['numpy','scipy','ml-dtypes','torch','torch-npu']:
   rows.append(dict(root=root,name=d.metadata['Name'],version=d.version,requires=d.requires))
out=pathlib.Path('/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_guarded_resume5/DEPENDENCY_REVIEW.json')
out.write_text(json.dumps({'packages':rows,'runtime_ABI':'unproven; metadata review only','conflict':'private numpy1.26.4 does not satisfy shared ml-dtypes0.6.0 numpy>=2'},indent=2))
print(out.read_text())
PY
