"""CPU-only dispatch test: slow CPU task must not idle NPU slot; in-flight <=2."""
import pathlib,tempfile,importlib.util,sys,json
P=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('ready',P/'tools/npu_pipeline_ready.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
with tempfile.TemporaryDirectory() as tmp:
 d=pathlib.Path(tmp);fake=d/'fake_guard.py';fake.write_text('''import sys,json,pathlib,subprocess
args=sys.argv[1:];receipt=pathlib.Path(args[args.index('--receipt')+1]);r=subprocess.run(args[args.index('--')+1:]);receipt.write_text(json.dumps({'phase':'finished','performance_valid':r.returncode==0,'workload_exit_code':r.returncode}));sys.exit(r.returncode)
''')
 jobs=[{'id':str(i),'prepare':{'cpu_only':True,'argv':[sys.executable,'-c','import time;time.sleep('+str(1.5 if i==0 else .05)+')']},'device':{'argv':[sys.executable,'-c','import time;time.sleep(.15)']}} for i in range(4)]
 assert m.run({'jobs':jobs,'continue_on_job_error':True},d/'run',lookahead=2,guard_command=[sys.executable,str(fake)])==0
 events=[json.loads(x) for x in (d/'run/events.jsonl').read_text().splitlines()];starts=[e['id'] for e in events if e['event']=='phase_start' and e['phase']=='device'];assert starts[0]=='1' and starts[-1]=='0',starts
 active=set();device=set();peak=0
 for e in events:
  if e['event']=='phase_start':
   active.add(e['id']);peak=max(peak,len(active))
   if e['phase']=='device':device.add(e['id']);assert len(device)==1
  if e['event']=='phase_end' and e['phase']=='device':active.remove(e['id']);device.remove(e['id'])
 assert peak<=2 and not active
 (P/'READY_OVERLAP_REGRESSION.json').write_text(json.dumps({'cpu_only':True,'passed':True,'device_order':starts,'peak_in_flight':peak,'events':events},indent=2));print('PASS ready order',starts,'in-flight limit',peak)
