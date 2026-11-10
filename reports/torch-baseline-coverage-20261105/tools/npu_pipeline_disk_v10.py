#!/usr/bin/env python3
"""Overlap verified CPU preparation with exclusive npu-guard device processes.

The JSON plan declares jobs with id, prepare, and device commands. A command is
{argv: [...], timeout: seconds, env: {...}}. Preparation must be independently
verified CPU-only and declared cpu_only=true. The device child owns all runtime
initialization and exits before the next device job. Never run this coordinator
inside another reservation. No device initialization occurs in this module.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import npu_guard as guard


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def validate(plan):
    jobs = plan['jobs']
    if not jobs or len({job['id'] for job in jobs}) != len(jobs):
        raise ValueError('Jobs must be nonempty with unique IDs')
    for job in jobs:
        if not isinstance(job['id'], str) or not job['id'] or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in job['id']):
            raise ValueError('IDs must contain letters, digits, underscores or hyphens')
        if job['prepare'].get('cpu_only') is not True:
            raise ValueError('Preparation requires an audited cpu_only=true declaration')
        for name in ('prepare', 'device'):
            cmd = job[name]
            if not isinstance(cmd['argv'], list) or not cmd['argv'] or not all(isinstance(x, str) and x for x in cmd['argv']):
                raise ValueError('Commands require nonempty string argv arrays')
            timeout = cmd.get('timeout', 120)
            if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
                raise ValueError('Phase timeout must be finite and positive')
            if not isinstance(cmd.get('env', {}), dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in cmd.get('env', {}).items()):
                raise ValueError('Environment must map strings to strings')
    wait = plan.get('wait', 600)
    if not isinstance(wait, (int, float)) or not math.isfinite(wait) or wait <= 0:
        raise ValueError('Guard wait must be finite and positive')


def run(plan, output, *, mode='pipeline', lookahead=2, guard_command=None):
    validate(plan)
    current = guard.LOCK_DIR / guard.LOCK_NAME
    try:
        reservation = json.loads(current.read_text())
    except (OSError, ValueError):
        reservation = {}
    if reservation.get('phase') == 'running' and reservation.get('child', {}).get('pid') == os.getpgrp():
        raise ValueError('Run the CPU coordinator outside npu-guard; device stages acquire their own reservations')
    if mode not in ('serial', 'pipeline') or lookahead < 1:
        raise ValueError('Invalid pipeline configuration')
    # The colocated guard implementation uses the SAME host lock and observer.
    guard_command = guard_command or [sys.executable, '-I', '-S', str(Path(guard.__file__).resolve()), 'run']
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'plan.json', plan)
    jobs = plan['jobs']
    states = [{'id': job['id'], 'status': 'pending'} for job in jobs]
    active = {}
    next_prepare = next_device = 0
    failure = None
    job_errors = []
    cancelled = []
    started = time.monotonic()
    journal = (output / 'events.jsonl').open('x')
    owned = guard.OwnedProcesses()

    def event(kind, **kw):
        journal.write(json.dumps(dict(event=kind, monotonic=time.monotonic(), wall=time.time(), **kw)) + '\n')
        journal.flush()

    def cancel(number, frame):
        cancelled.append(number)

    handlers = {s: signal.signal(s, cancel) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}

    def launch(index, phase):
        job = jobs[index]
        spec = job[phase]
        directory = output / job['id']
        directory.mkdir(exist_ok=True)
        env = {**os.environ, **plan.get('env', {}), **spec.get('env', {})}
        argv = spec['argv']
        if phase == 'device':
            argv = [*guard_command, '--owner', f'{plan.get("owner", "pipeline")}/{job["id"]}',
                    '--wait', str(plan.get('wait', 600)), '--timeout', str(spec.get('timeout', 120)),
                    '--receipt', str(directory / 'guard.json'), '--', *argv]
        log = (directory / f'{phase}.log').open('x')
        try:
            process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                       cwd=spec.get('cwd', plan.get('cwd')), env=env,
                                       start_new_session=(phase == 'prepare'))
        except BaseException:
            log.close()
            raise
        row = dict(process=process, log=log, index=index, phase=phase, start=time.monotonic())
        active['device' if phase=='device' else 'prepare:'+str(index)] = row
        states[index]['status'] = phase
        states[index][phase] = dict(start=row['start'], pid=process.pid)
        event('phase_start', id=job['id'], phase=phase, pid=process.pid)

    def stop_active():
        # CPU workers get their own managed process group; device groups remain
        # entirely under guard supervision, including its pidfd descendant drain.
        for key, row in active.items():
            phase=row['phase']
            proc = row['process']
            if phase == 'prepare':
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            elif proc.poll() is None:
                proc.terminate()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and any(r['process'].poll() is None for r in active.values()):
            time.sleep(.05)
        for key, row in active.items():
            phase=row['phase']
            proc = row['process']
            if phase == 'prepare':
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            # Do not SIGKILL the guard supervisor: it owns device cleanup.
            proc.wait()
            row['log'].close()
            state = states[row['index']]
            state[phase].update(end=time.monotonic(), returncode=proc.returncode)
            state['status'] = 'cancelled'
        active.clear()
        # Reap adopted CPU grandchildren as well. Direct Popen children have
        # already been waited, so the subreaper cannot swallow their exit codes.
        if owned.refresh():
            owned.send(signal.SIGTERM)
            deadline = time.monotonic() + 5
            while owned.refresh() and time.monotonic() < deadline:
                time.sleep(.05)
            if owned.refresh():
                owned.send(signal.SIGKILL)
                while owned.refresh():
                    time.sleep(.05)

    try:
        event('pipeline_start', mode=mode, compiler_workers=2, lookahead=lookahead)
        while any(s['status'] not in ('completed','failed') for s in states):
            if cancelled:
                failure = f'signal {cancelled[0]}'
                break
            for key, row in list(active.items()):
                phase=row['phase']
                proc = row['process']
                code = proc.poll()
                index = row['index']
                if phase == 'prepare' and time.monotonic()-row['start'] > jobs[index][phase].get('timeout', 120):
                    failure = f'{jobs[index]["id"]}: preparation timeout'
                    break
                if code is None:
                    continue
                if phase == 'prepare' and guard.group_alive(proc.pid):
                    failure = f'{jobs[index]["id"]}: preparation left background descendants'
                    break
                row['log'].close()
                del active[key]
                states[index][phase].update(end=time.monotonic(), returncode=code)
                event('phase_end', id=jobs[index]['id'], phase=phase, returncode=code)
                if code:
                    states[index]['status'] = 'failed'
                    message = f'{jobs[index]["id"]}: {phase} exit {code}'
                    if plan.get('continue_on_job_error'):
                        job_errors.append(message)
                        event('job_failure', id=jobs[index]['id'], reason=message)
                        continue
                    failure = message
                    break
                if phase == 'prepare':
                    states[index]['status'] = 'ready'
                else:
                    receipt = json.loads((output / jobs[index]['id'] / 'guard.json').read_text())
                    if receipt.get('phase') != 'finished' or receipt.get('performance_valid') is not True or receipt.get('workload_exit_code') != 0:
                        states[index]['status'] = 'invalid_guard'
                        failure = f'{jobs[index]["id"]}: invalid guard receipt'
                        break
                    states[index]['status'] = 'completed'
                    next_device += 1
            if failure:
                break
            ready=[i for i,s in enumerate(states) if s['status']=='ready']
            if 'device' not in active and ready:
                free=shutil.disk_usage(output).free/2**30
                admitted=[i for i in ready if free>=jobs[i].get('min_free_gib',50.5)]
                if admitted:launch(admitted[0], 'device')
                else:time.sleep(1)
            # Two total in-flight jobs, including running NPU, CPU and ready payloads.
            # If NPU is idle, both slots may prepare; an independent ready job need not wait for a slow earlier job.
            outstanding=sum(s['status'] in ('prepare','ready','device') for s in states)
            cpu_count=sum(r['phase']=='prepare' for r in active.values())
            can_prepare=(mode=='pipeline' or not active)
            while next_prepare<len(jobs) and can_prepare and outstanding<lookahead and cpu_count<2:
                launch(next_prepare,'prepare')
                next_prepare+=1;outstanding+=1;cpu_count+=1
                if mode=='serial':break
            if active:
                time.sleep(.025)
    except BaseException as exc:
        failure = f'{type(exc).__name__}: {exc}'
    finally:
        stop_active()
        owned.close()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        summary = dict(schema=1, mode=mode, compiler_workers=2, lookahead=lookahead,
                       start=started, end=time.monotonic(), jobs=states, error=failure, job_errors=job_errors,
                       completed=sum(row['status'] == 'completed' for row in states),
                       descendants_drained=True)
        summary['seconds'] = summary['end']-summary['start']
        summary['jobs_per_minute'] = 60 * summary['completed']/summary['seconds']
        write_json(output / 'summary.json', summary)
        event('pipeline_end', error=failure, completed=summary['completed'])
        journal.close()
    return (1 if job_errors else 0) if failure is None else (128+cancelled[0] if cancelled else 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--mode', choices=['serial', 'pipeline'], default='pipeline')
    parser.add_argument('--lookahead', type=int, default=2)
    args = parser.parse_args()
    return run(json.loads(args.plan.read_text()), args.out, mode=args.mode, lookahead=args.lookahead)


if __name__ == '__main__':
    sys.exit(main())
