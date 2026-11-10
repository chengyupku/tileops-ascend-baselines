"""Resume the existing final auditor only after independent chains have joined."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

HERE = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument('--log-version', required=True)
a = p.parse_args()


def read(name, default=None):
    try:
        return json.loads((HERE / name).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def state(stage, **extra):
    (HERE / 'completion-status.json').write_text(json.dumps(dict(stage=stage, **extra), indent=2))


def run(script, *args):
    subprocess.run(['python3', str(HERE / script), *args], check=True)


try:
    state('waiting_for_existing_acceptance_chains')
    while True:
        formal = read('formal-batch-v1-status.json', [])
        catlass = read('baseline-catlass-status.json', [])
        log = read('log-' + a.log_version + '-status.json', {})
        log1p = read('log1p-v6-status.json', {})
        assert log.get('stage') != 'stopped_for_review', log
        assert all(row['rc'] == 0 or row['label'].startswith('pytest-') for row in formal), formal
        assert all(row['rc'] == 0 for row in catlass), catlass
        if len(formal) == 28 and len(catlass) == 10 and log.get('stage') == 'complete' and log1p.get('stage') == 'complete':
            break
        time.sleep(5)
    state('checking_current_source_results')
    run('fetch.py', 'final', 'baseline', 'environment', 'edges', 'profiles')
    run('audit.py')
    audit = read('audit.json')
    assert audit['own_valid_cases'] == 20 and audit['cases_passing_including_additional'] == 20, audit['comparison']
    cmdline = Path('/proc/1263512/cmdline').read_bytes()
    assert b'/logroots4_20261028/finish_baselines.py' in cmdline, cmdline
    os.kill(1263512, signal.SIGCONT)
    state('existing_final_auditor_resumed')
    while True:
        progress = read('remaining-baselines-status.json', {})
        assert progress.get('stage') != 'stopped_for_review', progress
        if progress.get('stage') == 'ready_for_final_review':
            break
        time.sleep(5)
    run('review.py')
    state('ready_for_final_review')
except BaseException as exc:
    state('stopped_for_review', error=repr(exc))
    raise
