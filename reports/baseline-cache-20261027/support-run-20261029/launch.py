import pathlib, subprocess, json, sys, time, fcntl, os
host=sys.argv[1];root=pathlib.Path.cwd();campaign=root/'reports/baseline-cache-20261027';directory=campaign/'support-run-20261029';hostdir=campaign/host
with (hostdir/'supervisor.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (campaign/'STOP').exists() and not (hostdir/'STOP').exists(), 'Existing STOP must be reconciled'
cmd=['/usr/bin/python3','-u',str(root/'tools/run_baseline_support_batch_v1.py'),'--campaign',str(campaign),'--host',host,'--selection',str(directory/'SELECTION-v1.json')]
log=(directory/(host+'-supervisor.log')).open('ab')
proc=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
stat=pathlib.Path('/proc')/str(proc.pid)/'stat'
row={'at':time.time(),'pid':proc.pid,'start_ticks':int(stat.read_text().rsplit(')',1)[1].split()[19]),'boot_id':pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'cmd':cmd}
(directory/(host+'-LAUNCH.json')).write_text(json.dumps(row,indent=2)+'\n'); print(json.dumps(row))
