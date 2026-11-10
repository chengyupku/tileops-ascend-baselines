#!/usr/bin/env python3
"""Read-only server1 progress, queried from the remote using only stdlib."""
import argparse
import datetime
import json
import math
import subprocess
import sys
import time

REMOTE = r'''
import json,time,shutil
from pathlib import Path
h=Path('/opt/tileops-baseline-cache/workspace/reports/baseline-retest32-20261030/server1')
def read(p,default):
 try:return json.loads(p.read_text())
 except FileNotFoundError:return default
rows={}
for line in (h/'journal.jsonl').read_text().splitlines():
 try:r=json.loads(line);rows[r['id']]=r
 except ValueError:pass
assignment=read(h.parent/'server1_ASSIGNMENTS.json',{})['pairs']
state=read(h/'STATE.json',{});ident=state.get('identity',{});alive=False
try:
 fields=Path('/proc/'+str(ident['pid'])+'/stat').read_text().rsplit(')',1)[1].split()
 alive=int(fields[19])==ident['start_ticks'] and fields[0]!='Z'
except (FileNotFoundError,KeyError):pass
libraries={}
for t in assignment:
 lib=t['key'][0]; d=libraries.setdefault(lib,{'total':0,'cached':0,'pending':0,'untested':0})
 d['total']+=1;r=rows.get(t['id']);d['cached' if r and r['status']=='cached' else 'pending' if r else 'untested']+=1
active=[]
for j in read(h/'ACTIVE.json',{}).get('jobs',[]):
 if j['id'] in rows and rows[j['id']].get('batch')==read(h/'ACTIVE.json',{}).get('batch'):continue
 phase=read(Path(j['out'])/'phase.json',{}).get('phase','prepared_or_queued')
 active.append({'key':j['key'],'phase':phase})
print(json.dumps({'at':time.time(),'libraries':libraries,'assigned':len(assignment),'cached':sum(v['cached'] for v in libraries.values()),'pending':sum(v['pending'] for v in libraries.values()),'identity':ident,'alive':alive,'phase':state.get('phase'),'active':active,'disk_free_bytes':shutil.disk_usage(h).free,'disk_status':read(h/'DISK_STATUS.json',{}),'disk_policy':read(h/'DISK_POLICY.json',{}),'stop_requested':(h/'STOP').exists()}))
'''

def fetch():
    r=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ClearAllForwardings=yes','-o','ConnectTimeout=8','openlibing-DevEnv-661495','python3 -'],input=REMOTE,text=True,capture_output=True,timeout=20,check=True)
    return json.loads(r.stdout)

def render(d):
    done,total=d['cached'],d['assigned'];width=36;fill=width*done//total if total else 0
    free=d['disk_free_bytes']/1024**3;gate=d['disk_policy'].get('admission_free_bytes',50.5*1024**3)/1024**3
    stamp=datetime.datetime.fromtimestamp(d['at'],datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    lines=['一号远端 · baseline 32核补测',f"有效缓存 [{'█'*fill}{'░'*(width-fill)}] {done}/{total} ({100*done/total:.2f}%)",f"待处理 {d['pending']} | 未尝试 {total-done-d['pending']} | 未完成 {total-done}",f"磁盘空余 {free:.2f} GiB / {d['disk_free_bytes']/1e9:.2f} GB · 底线 50 GiB · 新任务门槛 {gate:.1f} GiB",f"磁盘状态：{'允许提交' if free>=gate else '暂停新任务，等待空间恢复'}",f"runner：{'存活且启动身份匹配' if d['alive'] else '未运行或身份不匹配'} · PID {d['identity'].get('pid','?')}"+(' · 当前批次收尾后停止/交接' if d['stop_requested'] else ''),'']
    for lib,v in d['libraries'].items():lines.append(f"{lib:28} {v['cached']:4}/{v['total']:<4}  待处理 {v['pending']}")
    lines+=['','当前批次（完整通过并入缓存才计入进度）：']
    lines += [f"  {r['key'][0]} / {r['key'][1]} / {r['key'][2]} · {r['phase']}" for r in d['active']] or ['  无未登记任务']
    lines+=['',f'远端采样：{stamp}；只读查询，不启动任务。']
    return '\n'.join(lines)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--interval',type=float,default=10);p.add_argument('--once',action='store_true');p.add_argument('--json',action='store_true');a=p.parse_args()
    if not math.isfinite(a.interval) or a.interval<2:p.error('--interval must be finite and >=2 seconds')
    try:
        while True:
            try:
                d=fetch();s=json.dumps(d,ensure_ascii=False,indent=2) if a.json else render(d)
            except (subprocess.SubprocessError,ValueError,OSError) as e:
                s=f'本次远端查询失败，不能确认当前进度：{e}'
            if sys.stdout.isatty() and not a.once and not a.json:print('\033[2J\033[H',end='')
            print(s,flush=True)
            if a.once or a.json:break
            time.sleep(a.interval)
    except KeyboardInterrupt:pass
if __name__=='__main__':main()
