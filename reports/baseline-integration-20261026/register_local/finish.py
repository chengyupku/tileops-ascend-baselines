from pathlib import Path
import json,hashlib,difflib
W=Path('/root/dyq/workspace');R=W/'reports/baseline-integration-20261026/register_local';P=W/'TileOPs/benchmarks/ascend_all_pairs/ops_nn'
p=P/'verified_runtime.py';s=p.read_text();s=s.replace("    vendor = Path(record['vendor']).resolve()", "    # Receipt pins include the unchanged vendor and original source dependencies.\n    checker = json.loads(Path(record['checker']).read_text())\n    workspace = HERE.parents[3]\n    for name, expected in checker['source_sha256'].items():\n        path = workspace / name\n        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:\n            raise UnsupportedCase('Reviewed dependency changed: ' + str(path))\n    vendor = Path(record['vendor']).resolve()")
p.write_text(s)
(P/'run_verified.py').write_text('''"""Run a normal baseline command in the reviewed local environment (inside npu-guard).

Usage: baseline-run ops-nn python run_verified.py OP CASE_ID WORKDIR -- COMMAND ...
The command may be the normal checker, benchmark, or another registered caller.
"""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('op')
    parser.add_argument('case_id')
    parser.add_argument('workdir', type=Path)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    records = json.loads((here / 'accepted_verified_20261026.json').read_text())['records']
    record = next((r for r in records if r['key'][1:] == [args.op, args.case_id]), None)
    if record is None:
        parser.error('Only the 16 reviewed exact keys are supported by this environment entry.')
    command = args.command
    if command[:1] == ['--']:
        command = command[1:]
    if not command:
        parser.error('A foreground command is required.')
    lock = json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text())
    if lock.get('phase') != 'running' or lock.get('child', {}).get('pid') != os.getpgrp():
        parser.error('The entire command must run inside npu-guard.')
    args.workdir.resolve().mkdir(parents=True, exist_ok=False)
    os.chdir(args.workdir.resolve())
    short = Path('/proc') / str(os.getpid()) / 'cwd'
    for name in ('tmp', 'ASCEND_CACHE_PATH', 'ASCEND_WORK_PATH', 'XDG_CACHE_HOME',
                 'TORCH_EXTENSIONS_DIR', 'ASCEND_PROCESS_LOG_PATH'):
        Path(name).mkdir()
        os.environ[name] = str(short / name)
    os.environ['TMPDIR'] = str(short / 'tmp')
    os.environ['OPS_NN_VENDOR'] = record['vendor']
    os.environ['ASCEND_CUSTOM_OPP_PATH'] = record['vendor']
    os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = '0'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    workspace = here.parents[3]
    os.environ['PYTHONPATH'] = os.pathsep.join([str(workspace / 'TileOPs'),
        str(workspace / 'TileOPs/src'), str(workspace), os.environ.get('PYTHONPATH', '')])
    os.execvp(command[0], command)


if __name__ == '__main__':
    main()
''')
# Rebuild precise patch/manifest against original backups, including newly created files.
changes=json.loads((R/'CHANGED_FILES.json').read_text());changes.append({'path':str(P/'run_verified.py'),'backup':None,'before_sha256':None})
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
patch=''
for row in changes:
 p=Path(row['path']);row['after_sha256']=sha(p);old=Path(row['backup']).read_text() if row['backup'] else ''
 patch+=''.join(difflib.unified_diff(old.splitlines(True),p.read_text().splitlines(True),fromfile=str(p.relative_to(W)) if row['backup'] else '/dev/null',tofile=str(p.relative_to(W))))
(R/'integration.patch').write_text(patch);(R/'CHANGED_FILES.json').write_text(json.dumps(changes,indent=2)+'\n')
