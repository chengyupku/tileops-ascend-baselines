"""CPU SSH orchestration; every device command is submitted through host guard."""
import argparse
import datetime
import json
from pathlib import Path
import shlex
import subprocess
import time

HOST = 'openlibing-DevEnv-661495'
ROOT = '/opt/tileops-all-pairs/workspace'
HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--label', required=True)
    p.add_argument('--timeout', type=int, default=600)
    p.add_argument('--wait', type=int, default=3600)
    p.add_argument('--library', default='tileops')
    p.add_argument('--env', default='env.sh')
    p.add_argument('--owner', default='logroots4-server1-20261028')
    p.add_argument('args', nargs=argparse.REMAINDER)
    a = p.parse_args()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    out = HERE / 'requests' / (a.label + '_' + stamp)
    out.mkdir(parents=True)
    args = a.args[1:] if a.args[:1] == ['--'] else a.args
    envfile = ROOT + '/docs/ascend950/dev/logroots4_20261028/' + a.env
    command = ['npu-guard','run','--owner',a.owner + '/' + a.label,'--wait',str(a.wait),
               '--timeout',str(a.timeout),'--','bash','-c',
               'source "$1"\nshift\nexec "$@"','arithmetic9',envfile,
               '/opt/tileops-ci/runtime/bin/python',*args]
    (out/'command.json').write_text(json.dumps(command,indent=2))
    begin = time.monotonic()
    acquired = None
    with (out/'guard.log').open('w') as f:
        r = subprocess.Popen(['ssh','-o','ClearAllForwardings=yes','-o','BatchMode=yes',HOST,shlex.join(command)],
                             stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in r.stdout:
            f.write(line);f.flush()
            if '[npu-guard] acquired device' in line:
                acquired = time.monotonic()
        r.wait()
    data = {'label':a.label,'started_utc':stamp,'wall_seconds':time.monotonic()-begin,
            'guard_exit_code':r.returncode,'library':a.library,'command':command,
            'completed_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    data['queue_and_idle_check_seconds'] = acquired - begin if acquired else None
    data['reservation_seconds'] = time.monotonic() - acquired if acquired else None
    (out/'status.json').write_text(json.dumps(data,indent=2))
    print(json.dumps({'request':str(out),**data}),flush=True)
    raise SystemExit(r.returncode)


if __name__ == '__main__': main()
