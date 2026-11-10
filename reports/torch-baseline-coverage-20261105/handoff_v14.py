import pathlib,json,time,os
p=pathlib.Path(__file__).resolve().parent
h=json.loads((p/'server1/HEARTBEAT.json').read_text());pid=h['pid'];start=str(h['start_ticks'])
assert h.get('status') in ('running','waiting_repairs')
(p/'server1/STOP').write_text('finish current batch, then v14 with additional current cases\n')
while True:
 proc=pathlib.Path('/proc')/str(pid)/'stat'
 if not proc.exists() or proc.read_text().split()[21]!=start:break
 time.sleep(2)
(p/'server1/STOP').unlink()
os.execv('/usr/bin/python3',['python3',str(p/'supervisor_server1_v14.py'),'server1'])
