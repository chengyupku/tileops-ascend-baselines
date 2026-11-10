import json,pathlib,time,argparse,collections
p=pathlib.Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--watch',action='store_true');a=ap.parse_args()
def read(f):
 try:return json.loads(f.read_text())
 except (FileNotFoundError,json.JSONDecodeError):return {}
while True:
 progress=read(p/'server1/PROGRESS.json');remote=read(p/'server1/REMOTE_STATUS.json')
 accepted=sum(x.get('accepted',False) for x in progress.values());failed=sum(not x.get('accepted',False) for x in progress.values())
 print(time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime()),f'accepted {accepted}/400 | archived pending {failed} | remote {remote.get("status")} | disk {remote.get("free_GiB",0):.2f} GiB',flush=True)
 if not a.watch:break
 time.sleep(10)
