#!/usr/bin/env python3
"""Serial, restartable per-host cache sweep. CPU supervisor never owns the NPU.

One fresh guarded process per pair. No courtesy delays, failure fallback, or
automatic numerical retries. Performance cache requires guard AND full checker.
"""
import argparse
import collections
import copy
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from baseline_cache_guard_binding import capture_guard, bind_guard

ROOT = Path(__file__).resolve().parents[1]


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def put(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, path)


def append(path, value):
    with open(path, 'a') as f:
        f.write(json.dumps(value, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def merge_quarantine_delta(path, before, after):
    """Publish new automatic holds without restoring a stale control snapshot.

    A controller can add, release, or revise holds while a device pair runs.
    Only this pair's actual changes may be merged; newer control values win.
    Control writers use the same QUARANTINE.lock when editing the file.
    """
    path = Path(path)
    delta = {k: v for k, v in after.items() if k not in before or v != before[k]}
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        current = read(path, {})
        changed = False
        for key, value in delta.items():
            if key not in before:
                if key not in current:
                    current[key] = value
                    changed = True
            elif current.get(key) == before[key]:
                current[key] = value
                changed = True
        if changed:
            put(path, current)
        return current


def identity(pid):
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': pid, 'start_ticks': int(fields[19]), 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}


def quarantine_key(task, quarantine):
    exact = 'pair:' + str(task.get('id', ''))
    if exact in quarantine:
        return exact
    library, op, _ = task['key']
    scoped = library + '::' + op
    return scoped if scoped in quarantine else library if library in quarantine else None


def assigned_history(tasks, rows, cache_rows):
    """Keep migrated ownership out of this host's live counters; retain journals."""
    assigned = {t['id'] for t in tasks}
    latest = {x['id']: x for x in rows if x['id'] in assigned}
    cached_ids = {x['task_id'] for x in cache_rows if x['task_id'] in assigned}
    return latest, cached_ids


def completed_ids(latest, retry, quarantine, cached_ids):
    """Reconsider released skips, but never repeat a durable successful cache."""
    done = {k for k, x in latest.items() if x['status'] != 'resource_wait' and
            (x['status'] != 'temporarily_skipped_library' or quarantine_key(x, quarantine)) and
            (x['status'] == 'cached' or k not in retry or x.get('repair_revision') == retry[k]['revision'])}
    return done | cached_ids


def record_deployment_failure(quarantine, failures, task, out):
    """An operator's missing dependency cannot suppress the whole library.

    Explicit evidence-backed library holds remain available. Count distinct
    failed pairs, not repetitions of a resource gate.
    """
    scope = '::'.join(task['key'][:2])
    failed = failures.setdefault(scope, {})
    failed[task['id']] = str(out)
    if len(failed) >= 3:
        quarantine.setdefault(scope, {
            'reason': 'three operator-scoped deployment failures; inspect before retry',
            'scope': scope, 'out': str(out), 'failed_pairs': dict(failed)})


def record_timeout(quarantine, task, out):
    """A single bounded timeout establishes a hold for that exact pair only.

    Host health remains enforced independently by npu-guard before every run.
    Broader holds require separately inspected evidence; queue waits never call
    this function. Retain the failed attempt rather than retrying it unchanged.
    """
    quarantine.setdefault('pair:' + task['id'], {
        'reason': 'bounded timeout or device synchronization failure; exact pair held for inspection',
        'first_pair': task['key'], 'out': str(out), 'scope': 'exact_pair'})


def classify_failure(code, measurement, guard):
    if guard.get('conflict_detected') or code == 76:
        return 'performance_invalid_external_conflict'
    if code in (75, 78) and not measurement:
        return 'resource_wait'
    if measurement.get('status') == 'correctness_not_verified':
        return 'correctness_not_verified'
    return 'failed'


def live_external_blockers(path):
    """Only wait for explicitly observed competing batch process identities.

    This is a resource precondition, not a time-based courtesy window. A PID
    reused by a different process cannot extend the wait. npu-guard still
    independently checks the physical device after these batches terminate.
    """
    current = []
    config = read(path, {})
    for entry in config.get('processes', []):
        try:
            if identity(entry['pid']) == {k:entry[k] for k in ('pid','start_ticks','boot_id')}:
                fields = Path(f'/proc/{entry["pid"]}/stat').read_text().rsplit(')',1)[1].split()
                if fields[0] != 'Z':
                    current.append(entry)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    # A confirmed external driver batch may be restarted with a new PID. Match
    # only the explicitly recorded user/program argument, never all user jobs.
    for rule in config.get('batch_rules', []):
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():
                continue
            try:
                if proc.stat().st_uid != rule['uid']:
                    continue
                args = (proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
                if not args or Path(args[0]).name not in ('python','python3','python3.12') or rule['program_argument'] not in args:
                    continue
                native = identity(int(proc.name))
                if (proc/'stat').read_text().rsplit(')',1)[1].split()[0] != 'Z':
                    current.append({**native,'uid':rule['uid'],'program_argument':rule['program_argument'],
                                    'description':rule['evidence']})
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
    return current


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--host', choices=['local', 'server1', 'server2'], required=True)
    p.add_argument('--limit', type=int)
    a = p.parse_args()
    campaign = a.campaign.resolve()
    hostdir = campaign / a.host
    hostdir.mkdir(parents=True, exist_ok=True)
    lock = (hostdir / 'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    tasks = [x for x in read(campaign / 'PLAN.json')['pairs'] if x['host'] == a.host]
    journal = hostdir / 'journal.jsonl'
    rows = [json.loads(x) for x in journal.read_text().splitlines()] if journal.exists() else []
    cache_rows = [json.loads(x) for x in (hostdir/'CACHE.jsonl').read_text().splitlines()] if (hostdir/'CACHE.jsonl').exists() else []
    latest, cached_ids = assigned_history(tasks, rows, cache_rows)
    retry = read(hostdir / 'RETRY.json', {})
    quarantine = read(hostdir / 'QUARANTINE.json', {})
    done = completed_ids(latest, retry, quarantine, cached_ids)
    failures = {}
    started = time.time()
    pid = identity(os.getpid())
    counts = collections.Counter(x['status'] for k,x in latest.items() if k in done)

    def live(phase, **extra):
        put(hostdir / 'LIVE.json', {'at': time.time(), 'started_at': started, 'host': a.host,
            'identity': pid, 'phase': phase, 'assigned': len(tasks), 'terminal': len(done),
            'remaining': len(tasks) - len(done), 'counts': dict(counts), 'quarantine': quarantine, **extra})

    executed = 0
    for task in tasks:
        if task['id'] in done:
            continue
        if (campaign / 'STOP').exists() or (hostdir / 'STOP').exists():
            live('stopped_by_control')
            return
        library = task['key'][0]
        quarantine = read(hostdir / 'QUARANTINE.json', {})
        blocked = quarantine_key(task, quarantine)
        if blocked:
            row = {'id': task['id'], 'key': task['key'], 'status': 'temporarily_skipped_library',
                   'reason': quarantine[blocked], 'quarantine_scope': blocked, 'at': time.time()}
        else:
            wait_started = time.time()
            while True:
                blockers = live_external_blockers(hostdir / 'EXTERNAL_BLOCKERS.json')
                if not blockers:
                    break
                live('waiting_external_batch', key=task['key'], external_blockers=blockers,
                     external_wait_seconds=time.time()-wait_started,
                     device_command_started=False, owns_device_lock=False)
                if (campaign / 'STOP').exists() or (hostdir / 'STOP').exists():
                    live('stopped_by_control')
                    return
                time.sleep(5)
            out = hostdir / 'pairs' / task['id']
            out.mkdir(parents=True, exist_ok=True)
            # Never overwrite a partial previous attempt or mistake it for success.
            if (out / 'task.json').exists():
                out = out / ('resume-' + str(time.time_ns()))
                out.mkdir()
            put(out / 'task.json', task)
            owner = f'baseline-cache-20261027/{a.host}/{task["id"]}'
            cmd = ['npu-guard', 'run', '--owner', owner, '--wait', '3600', '--timeout', str(task.get('timeout', 600)), '--on-conflict', 'terminate', '--',
                   'bash', str(ROOT / 'tools/baseline_cache_python.sh'), library,
                   str(ROOT / 'tools/measure_baseline_cache_pair.py'), '--task', str(out / 'task.json'), '--out', str(out)]
            before = time.time()
            guard_before = capture_guard()
            with (out / 'run.log').open('w') as log:
                proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
                child = identity(proc.pid)
                while proc.poll() is None:
                    metadata = read('/run/lock/ascend-npu/device-0.lock', {})
                    phase = read(out / 'phase.json', {})
                    live('running', key=task['key'], out=str(out), guard_identity=child,
                         guard_phase=metadata.get('phase') if metadata.get('owner') == owner else 'waiting_for_device',
                         pair_phase=phase, pair_elapsed_seconds=time.time()-before)
                    time.sleep(5)
            guard = read('/run/lock/ascend-npu/device-0.lock', {})
            put(out / 'guard.json', guard)
            measurement = read(out / 'measurement.json', {})
            valid = (proc.returncode == 0 and measurement.get('status') == 'measured_pending_guard'
                     and guard.get('owner') == owner and guard.get('performance_valid') is True
                     and guard.get('phase') == 'finished' and guard.get('workload_exit_code') == 0)
            status = 'cached' if valid else classify_failure(proc.returncode, measurement, guard)
            row = {'id': task['id'], 'key': task['key'], 'status': status, 'at': time.time(),
                   'wall_seconds': time.time()-before, 'exit_code': proc.returncode,
                   'measurement': str(out / 'measurement.json'), 'guard': str(out / 'guard.json'),
                   'error': measurement.get('error'), 'measurement_status': measurement.get('status')}
            if valid:
                measurement.update(status='cached', performance_valid=True, guard_receipt=str(out / 'guard.json'))
                try:
                    measurement['guard_binding'] = bind_guard(measurement, guard_before, capture_guard())
                except (OSError, ValueError) as exc:
                    measurement.update(status='guard_provenance_failed', performance_valid=False, error=str(exc))
                    row.update(status='guard_provenance_failed', error=str(exc))
                    put(out / 'measurement.json', measurement)
                    append(journal, row)
                    done.add(task['id'])
                    counts[row['status']] += 1
                    live('between_pairs')
                    continue
                put(out / 'measurement.json', measurement)
                append(hostdir / 'CACHE.jsonl', measurement)
                publish_best(hostdir)
            else:
                quarantine_before = copy.deepcopy(quarantine)
                log_tail = (out / 'run.log').read_text(errors='replace')[-12000:]
                error = str(measurement.get('error', '')) + log_tail
                hang = proc.returncode == 124 or measurement.get('status') == 'device_sync_failed' or guard.get('termination_reason') == 'timeout'
                deployment = any(x in error for x in ('ModuleNotFoundError', 'FileNotFoundError', 'cannot open shared object',
                    'Load the isolated original vendor', 'compiler route retired', 'No module named',
                    'deployment_blocked', 'Missing library:', 'No such file or directory'))
                if hang:
                    record_timeout(quarantine, task, out)
                elif deployment:
                    record_deployment_failure(quarantine, failures, task, out)
                quarantine = merge_quarantine_delta(hostdir / 'QUARANTINE.json', quarantine_before, quarantine)
                if status == 'performance_invalid_external_conflict':
                    append(hostdir / 'DEFERRED.jsonl', {**row, 'retry_condition': 'external competing workload ended; record changed condition before retry'})
                if status == 'resource_wait':
                    # Queue time is not a numerical/library failure or a reason
                    # to churn workers. Keep this process and retry the resource gate.
                    append(journal, row)
                    counts[status] += 1
                    live('resource_wait', key=task['key'])
                    return 75
            executed += 1
        if task['id'] in retry:
            row['repair_revision'] = retry[task['id']]['revision']
            row['repair_reason'] = retry[task['id']]['reason']
        append(journal, row)
        done.add(task['id'])
        counts[row['status']] += 1
        live('between_pairs')
        if a.limit and executed >= a.limit:
            live('pilot_complete')
            return
    live('sweep_complete')


def publish_best(hostdir):
    best = {}
    rows = [json.loads(x) for x in (hostdir / 'CACHE.jsonl').read_text().splitlines()]
    for r in rows:
        # Never compare across different host/software cohorts or timing modes.
        key = (r['key'][1], r['key'][2], r['cohort_key'], r['comparison_metric'])
        latency = r['timing']['latency_ms'] if r['comparison_metric'] == 'device_latency_ms' else r['end_to_end_mean_ms']
        if key not in best or latency < best[key]['latency_ms']:
            best[key] = {'op': key[0], 'case_id': key[1], 'cohort': key[2], 'metric': key[3],
                         'library': r['key'][0], 'latency_ms': latency, 'cache_key': r['cache_key'],
                         'coverage': 'best_among_successfully_measured_candidates_only'}
    put(hostdir / 'BEST.json', list(best.values()))


if __name__ == '__main__':
    while True:
        result = main()
        if result != 75:
            raise SystemExit(result)
        time.sleep(30)
