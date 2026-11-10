#!/usr/bin/env python3
"""二号远端只读进度条；单条持久 SSH，每秒刷新。"""
import argparse
import datetime
import json
import math
import select
import subprocess
import sys
from baseline_retest32_server1_dashboard import REMOTE as TEMPLATE

REMOTE = TEMPLATE.replace('/opt/tileops-baseline-cache/workspace', '/opt/tileops-all-pairs/workspace').replace('server1', 'server2')

def bar(done, total, width=32):
    n = min(width, width * done // total) if total else 0
    return f"[{'█' * n}{'░' * (width - n)}] {done}/{total} ({100 * done / total if total else 0:.2f}%)"

def render(d):
    total, done = d['assigned'], d['cached']
    lines = ['二号远端 · baseline 32核补测', '有效缓存 ' + bar(done, total),
             f"待诊断 {d['pending']} | 未尝试 {total - done - d['pending']} | 尚未缓存 {total - done}",
             f"runner：{'存活且身份匹配' if d['alive'] else '未运行或身份不匹配'} · PID {d['identity'].get('pid', '?')}" + (' · 已请求批次收尾后停止' if d['stop_requested'] else ''),
             f"磁盘空余 {d['disk_free_bytes'] / 1024**3:.1f} GiB", '']
    for lib, v in d['libraries'].items():
        lines.append(f"{lib:18} {bar(v['cached'], v['total'], 20)}  待诊断 {v['pending']}")
    lines += ['', '当前批次：']
    lines += [f"  {r['key'][0]} / {r['key'][1]} / {r['key'][2]} · {r['phase']}" for r in d['active']] or ['  无未登记任务']
    stamp = datetime.datetime.fromtimestamp(d['at'], datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    lines += ['', f'远端采样 {stamp} · 单条持久SSH · Ctrl+C退出', '仅完整校验、有效性能且已入缓存才计入进度；批次收取前不会提前计数。']
    return '\n'.join(lines)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--interval', type=float, default=1)
    p.add_argument('--once', action='store_true')
    p.add_argument('--json', action='store_true', help='输出一次JSON快照')
    a = p.parse_args()
    if not math.isfinite(a.interval) or a.interval < 1:
        p.error('--interval must be finite and >=1 second')
    once = a.once or a.json
    script = 'import time\nwhile True:\n try:\n  exec(' + repr(REMOTE) + ')\n except Exception as e:\n  import json\n  print(json.dumps({"error":str(e)}))\n'
    script += ' break\n' if once else f' time.sleep({a.interval!r})\n'
    proc = subprocess.Popen(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ClearAllForwardings=yes', '-o', 'ConnectTimeout=8', '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=2', 'openlibing-DevEnv-055409', 'python3 -u -'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        proc.stdin.write(script)
        proc.stdin.close()
        while True:
            if not select.select([proc.stdout], [], [], max(30, a.interval + 20))[0]:
                raise RuntimeError('远端采样超时，当前进度未知')
            line = proc.stdout.readline()
            if not line:
                if once and proc.wait() == 0:
                    break
                raise RuntimeError('SSH已断开，当前进度未知；请重新运行')
            d = json.loads(line)
            if 'error' in d:
                raise RuntimeError(d['error'])
            if sys.stdout.isatty() and not once:
                print('\033[2J\033[H', end='')
            print(json.dumps(d, ensure_ascii=False, indent=2) if a.json else render(d), flush=True)
            if once:
                break
    except KeyboardInterrupt:
        pass
    except (OSError, ValueError, RuntimeError) as e:
        print(f'查询失败：{e}', file=sys.stderr)
        return 1
    finally:
        if proc.poll() is None:
            proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
