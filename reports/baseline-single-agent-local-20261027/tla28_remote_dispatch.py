"""Foreground SSH transport for this agent's own sequential exact-case command."""
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
REMOTE = '/opt/tileops-all-pairs/workspace/reports/baseline-single-agent-local-20261027-server2'
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
       '-o', 'ControlPath=/tmp/allpairs-server2-20261022.sock', 'openlibing-DevEnv-055409']


def run_remote(attempt, timeout, case_args):
    assert timeout <= 120 and attempt.startswith('tla28-')
    out = HERE/attempt
    out.mkdir(exist_ok=True)
    assert not (out/'guard.json').exists()
    cmd = ['python3', REMOTE+'/remote_guard.py', attempt, str(timeout),
           'python3', REMOTE+'/task_exec.py', '/opt/tileops-all-pairs/bin/allpairs-python',
           REMOTE+'/run_tla28_remote_case.py', attempt, *case_args]
    (out/'TRANSPORT.json').write_text(json.dumps({'host':SSH[-1], 'command':cmd,
        'remote_directory':REMOTE+'/'+attempt, 'foreground':True}, indent=2)+'\n')
    with (out/'ssh.log').open('w') as log:
        proc = subprocess.run([*SSH, shlex.join(cmd)], stdout=log, stderr=subprocess.STDOUT,
                              timeout=330)
    # Keep device/runtime logs at the remote path; retrieve compact exact evidence.
    code = ('import pathlib,tarfile,sys; p=pathlib.Path('+repr(REMOTE)+'); '
            't=tarfile.open(fileobj=sys.stdout.buffer,mode="w|gz"); '
            '[t.add(f,arcname=str(f.relative_to(p)),recursive=False) '
            'for f in (p/'+repr(attempt)+').iterdir() '
            'if f.is_file() and f.suffix in (".json",".log",".txt",".cu")]; t.close()')
    archive = out/'evidence.tar.gz'
    with archive.open('wb') as f:
        transfer = subprocess.run([*SSH, shlex.join(['python3','-c',code])],
                                  stdout=f, stderr=subprocess.PIPE, timeout=60)
    if transfer.returncode:
        raise RuntimeError('Evidence transfer failed: '+transfer.stderr.decode()[-400:])
    with tarfile.open(archive,'r:gz') as tf:
        for member in tf.getmembers():
            path = Path(member.name)
            assert member.isfile() and path.parts[0] == attempt and '..' not in path.parts
            target = HERE/path
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(tf.extractfile(member).read())
    receipt = json.loads((out/'guard.json').read_text())
    receipt['host'] = SSH[-1]
    receipt['remote_directory'] = REMOTE+'/'+attempt
    (out/'guard.json').write_text(json.dumps(receipt,indent=2)+'\n')
    log = (out/'guard.log').read_text(errors='replace')
    if not (out/'checker.json').exists() and any(s in log for s in ('FileNotFoundError:', 'ModuleNotFoundError:', 'ImportError:')):
        route_file = HERE/'TLA28_REMOTE_ROUTE.json'
        route = json.loads(route_file.read_text())
        route.update(ready=False, startup_failure=attempt)
        temp = route_file.with_suffix('.tmp')
        temp.write_text(json.dumps(route,indent=2)+'\n')
        temp.replace(route_file)
    return receipt['returncode']
