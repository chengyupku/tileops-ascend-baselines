#!/usr/bin/env python3
import json,pathlib,time,argparse
p=pathlib.Path(__file__).resolve().parents[1]/'reports/baseline-flex-20261106'
a=argparse.ArgumentParser();a.add_argument('--watch',action='store_true');args=a.parse_args()
while True:
 total=ok=0
 for host in ['server1','server2']:
  h=p/host
  if not (h/'PLAN.json').exists():print(host,'not dispatched');continue
  rows=json.load(open(h/'PLAN.json'))['rows'];state=json.load(open(h/'STATE.json')) if (h/'STATE.json').exists() else {};pr=json.load(open(h/'PROGRESS.json')) if (h/'PROGRESS.json').exists() else {};n=sum(pr.get(r['id'],{}).get('status')=='accepted' for r in rows);total+=len(rows);ok+=n;print(host,f'{n}/{len(rows)}',state.get('phase'),state.get('current'))
 print(f'Accepted {ok}/{total}; full raw replay required',flush=True)
 if not args.watch:break
 time.sleep(10)
