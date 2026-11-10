#!/usr/bin/env python3
"""Read-only local progress, refreshed by the background controller."""
import argparse,json,pathlib,time,os
P=pathlib.Path(__file__).resolve().parents[1]/'reports/baseline-v3-full-20261102/server1'
def read(name):
 try:return json.loads((P/name).read_text())
 except (FileNotFoundError,json.JSONDecodeError):return {}
def show():
 s=read('STATE.json');p=read('PROGRESS.json');d=read('DISK_STATUS.json');ok=sum(x['status']=='accepted' for x in p.values());bad=sum(x['status']=='pending' for x in p.values());total=2814
 print('一号远端 baseline v3 + 完整 raw')
 print('['+'#'*int(ok/total*40)+'-'*(40-int(ok/total*40))+f'] {ok}/{total} ({ok/total:.2%})')
 print(f'待处理 {bad} | 未验收 {total-ok} | 阶段 {s.get("phase","未启动")}')
 print('当前:', ' / '.join(s.get('current',[])))
 print('磁盘:',round(d.get('free_GiB',0),2),'GiB | 磁盘采样距今',round(time.time()-d.get('at',time.time())),'秒')
 print('状态更新距今:',round(time.time()-s.get('at',time.time())),'秒；完成数仅计 checker、guard、raw复算、归档校验均通过')
 if bad:
  print('最近待处理:',next((x['key'] for x in reversed(list(p.values())) if x['status']=='pending'),None))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--watch',action='store_true');args=a.parse_args()
 while True:
  if args.watch:print('\033[2J\033[H',end='')
  show()
  if not args.watch:break
  time.sleep(5)
