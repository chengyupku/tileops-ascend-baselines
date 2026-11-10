"""CPU-only boundary handoff; never signals an active device task."""
import pathlib,json,time,os,sys
root=pathlib.Path('/root/dyq/workspace');b=root/'reports/baseline-cache-20261027';h=b/'local';r=b/'support-run-20261029';record=json.load(open(r/'INDEX_HANDOFF_V17.json'));stop=h/'STOP'
def same():
 try:
  p=pathlib.Path('/proc')/str(record['previous_identity']['pid']);s=(p/'stat').read_text().rsplit(')',1)[1].split()
  return int(s[19])==record['previous_identity']['start_ticks'] and s[0]!='Z'
 except FileNotFoundError:return False
while same():time.sleep(1)
if not stop.exists() or stop.read_text()!=record['token']:raise RuntimeError('STOP ownership changed; do not restart')
stop.unlink();record['restarted_at']=time.time();(r/'INDEX_HANDOFF_V17.json').write_text(json.dumps(record,indent=2))
import pwd
account=pwd.getpwuid(os.getuid())
os.environ['HOME']=account.pw_dir
os.environ['USER']=account.pw_name
os.environ['LOGNAME']=account.pw_name
os.chdir(root);os.execv('/usr/bin/python3',['/usr/bin/python3','-u',str(root/'tools/run_baseline_support_batch_index_v16.py'),'--campaign',str(b),'--host','local','--selection',str(r/'QUEUE.json')])
