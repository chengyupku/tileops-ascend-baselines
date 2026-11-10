"""Collect the complete experiment while retaining an unmet performance gate."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

HERE = Path(__file__).resolve().parent


def read(name, default=None):
    try:
        return json.loads((HERE / name).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def state(stage, **extra):
    (HERE / 'deferred-finalization-status.json').write_text(json.dumps(dict(stage=stage, **extra), indent=2))


def run(script, *args):
    subprocess.run(['python3', str(HERE / script), *args], check=True)


try:
    state('waiting_for_all_existing_validation_and_baseline_jobs')
    while True:
        formal = read('formal-batch-v1-status.json', [])
        catlass = read('baseline-catlass-status.json', [])
        log = read('log-v7-validation-status.json', {})
        log1p = read('log1p-v6-status.json', {})
        assert log.get('stage') != 'stopped_for_review', log
        assert all(row['rc'] == 0 or row['label'].startswith('pytest-') for row in formal), formal
        assert all(row['rc'] == 0 for row in catlass), catlass
        if len(formal) == 28 and len(catlass) == 10 and log.get('stage') == 'complete' and log1p.get('stage') == 'complete':
            break
        time.sleep(5)
    state('auditing_complete_experiment_with_retained_performance_failure')
    run('fetch.py', 'final', 'baseline', 'environment', 'edges', 'profiles', 'search')
    run('audit.py')
    run('review.py')
    run('timing_report.py')
    audit = read('audit.json')
    review = read('final-review.json')
    assert audit['own_valid_cases'] == 20, audit
    assert audit['formal_valid_baseline_pairs'] == 85 and audit['additional_valid_baseline_pairs'] == 3, audit
    assert all(review['sources_match_frozen'].values()), review
    assert all(v['edges']['qualified'] and v['profile']['qualified'] for v in review['operators'].values()), review
    # The original success-only coordinator has no child and was deliberately
    # paused. Stop that CPU waiter, rather than bypass its 20/20 performance gate.
    cmdline = Path('/proc/1263512/cmdline').read_bytes()
    assert b'/logroots4_20261028/finish_baselines.py' in cmdline, cmdline
    os.kill(1263512, signal.SIGTERM)
    os.kill(1263512, signal.SIGCONT)
    result = {
        'status': 'experiment_complete_with_deferred_performance_case',
        'all_performance_targets_met': audit['cases_passing_including_additional'] == 20,
        'source_manifest': 'frozen-v7.json',
        'correct_current_source_cases': audit['own_valid_cases'],
        'formal_valid_baseline_pairs': audit['formal_valid_baseline_pairs'],
        'additional_valid_baseline_pairs': audit['additional_valid_baseline_pairs'],
        'passing_performance_cases': audit['cases_passing_including_additional'],
        'minimum_ratio': audit['minimum_ratio_including_additional'],
        'comparison': audit['comparison'],
        'validation': review['operators'],
        'timing': read('timing-report.json'),
        'deferred_reason': 'Log BF16 16M remains below 0.8 after the bounded search. All failing, invalid and rejected experiments are retained; no passing search score replaces formal M4.',
        'evidence': ['audit.json', 'comparison.csv', 'final-review.json', 'qualified-baselines.json', 'log-v7-status.json', 'log-v7-validation-status.json', 'public-lut-v5-status.json'],
        'memory_branch': '.agent_memory/THREADS/logroots4-server1-20261028',
        'main_memory_changed': False,
    }
    (HERE / 'FINAL.json').write_text(json.dumps(result, indent=2) + '\n')
    state('ready_for_final_review', performance_passes=result['passing_performance_cases'], minimum_ratio=result['minimum_ratio'])
except BaseException as exc:
    state('stopped_for_review', error=repr(exc))
    raise
