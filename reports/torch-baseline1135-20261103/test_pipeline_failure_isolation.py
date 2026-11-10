"""CPU orchestration regression only; mock guard is never used by campaign jobs."""
import pathlib,tempfile,importlib.util,sys,json
P=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('campaign_pipeline',P/'tools/npu_pipeline_campaign.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
with tempfile.TemporaryDirectory() as tmp:
 d=pathlib.Path(tmp);fake=d/'fake_guard.py'
 fake.write_text('''import sys,json,pathlib,subprocess
args=sys.argv[1:];receipt=pathlib.Path(args[args.index('--receipt')+1]);command=args[args.index('--')+1:];r=subprocess.run(command);receipt.write_text(json.dumps({'phase':'finished','performance_valid':r.returncode==0,'workload_exit_code':r.returncode}));sys.exit(r.returncode)
''')
 jobs=[]
 for i in range(3):
  prepare='import time;time.sleep(.1)'+(';raise RuntimeError("expected preparation failure")' if i==1 else '')
  device='import time,pathlib;time.sleep(.5);pathlib.Path('+repr(str(d/('device'+str(i))))+').touch()'
  jobs.append({'id':str(i),'prepare':{'cpu_only':True,'argv':[sys.executable,'-c',prepare]},'device':{'argv':[sys.executable,'-c',device]}})
 rc=m.run({'jobs':jobs,'continue_on_job_error':True},d/'run',lookahead=2,guard_command=[sys.executable,str(fake)])
 s=json.loads((d/'run/summary.json').read_text());assert rc==1 and [x['status'] for x in s['jobs']]==['completed','failed','completed'],s
 assert (d/'device0').exists() and (d/'device2').exists() and not (d/'device1').exists()
 (P/'PIPELINE_REGRESSION.json').write_text(json.dumps({'cpu_only':True,'passed':True,'summary':s},indent=2));print('PASS: prepare failure preserved active device job and continued next valid job')
