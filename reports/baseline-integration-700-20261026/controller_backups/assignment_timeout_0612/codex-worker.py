#!/usr/bin/env python3
"""Cross-home Codex workers: bounded concurrency, supervised execution, durable inbox."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import importlib.util
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time
import tomllib
import uuid
_peer_spec = importlib.util.spec_from_file_location('worker_peer_mail', Path(__file__).with_name('worker_peer_mail.py'))
worker_peer_mail = importlib.util.module_from_spec(_peer_spec)
_peer_spec.loader.exec_module(worker_peer_mail)

BASE = Path('/root/dyq')
ROOT = Path(os.environ.get('CODEX_WORKER_ROOT', BASE/'codex-workers'))
BINARY = BASE/'.local/lib/codex/bin/codex'
SCRIPT = Path(__file__).resolve()
LIMIT = 8
ACTIVE = ('starting', 'running', 'stopping')


class WorkerError(Exception):
    pass


def registry():
    return json.loads((BASE/'agent-bin/homes.json').read_text())['codex']


def home_named(name):
    name = 'codex-0' if name == 'codex' else name
    for home in registry():
        if home['name'] == name or Path(home['path']).resolve() == Path(name).resolve():
            return home
    raise WorkerError('Unknown home: '+name)


def environment(home):
    env = os.environ.copy()
    env['CODEX_HOME'] = home['path']
    env['CODEX_SQLITE_HOME'] = str(BASE/'.codex')
    for key in ('CODEX_THREAD_ID', 'CODEX_PARENT_THREAD_ID', 'OPENAI_API_KEY',
                'CODEX_API_KEY', 'CODEX_ACCESS_TOKEN', 'OPENAI_FEDERATION_RULE_ID',
                'OPENAI_IDENTITY_TOKEN_FILE'):
        env.pop(key, None)
    return env


@contextmanager
def database():
    ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    db = sqlite3.connect(ROOT/'workers.sqlite', timeout=15, isolation_level=None)
    db.row_factory = sqlite3.Row
    try:
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, parent TEXT NOT NULL, '
                   'status TEXT NOT NULL, data TEXT NOT NULL, ack INTEGER NOT NULL DEFAULT 0)')
        db.execute('CREATE INDEX IF NOT EXISTS jobs_parent ON jobs(parent, status)')
        db.execute('BEGIN IMMEDIATE')
        yield db
        db.commit()
    except BaseException:
        db.rollback()
        raise
    finally:
        db.close()


def decode(row):
    job = json.loads(row['data'])
    job.update(id=row['id'], parent=row['parent'], status=row['status'], ack=bool(row['ack']))
    return job


def save(db, job):
    db.execute('UPDATE jobs SET status=?, data=?, ack=? WHERE id=?',
               (job['status'], json.dumps(job, ensure_ascii=False), int(job.get('ack', False)), job['id']))


def get_job(ident):
    with database() as db:
        row = db.execute('SELECT * FROM jobs WHERE id=?', (ident,)).fetchone()
    if row is None:
        raise WorkerError('Unknown job: '+ident)
    return decode(row)


def update(ident, **changes):
    with database() as db:
        job = decode(db.execute('SELECT * FROM jobs WHERE id=?', (ident,)).fetchone())
        job.update(changes)
        save(db, job)
    return job


def process_stamp(pid):
    """PID + kernel starttime + boot id prevents signaling a reused process id."""
    try:
        stat = Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')', 1)[1].split()
        if stat[0] == 'Z':
            return None
        return Path('/proc/sys/kernel/random/boot_id').read_text().strip()+':'+stat[19]
    except (OSError, ValueError, TypeError):
        return None


def alive(job, prefix='supervisor'):
    stamp = job.get(prefix+'_stamp')
    return bool(stamp and process_stamp(job.get(prefix+'_pid')) == stamp)


def group_members(job):
    """Find remaining members even after the child group leader has exited."""
    stamp = job.get('child_stamp')
    pid = job.get('child_pid')
    if not stamp or not pid:
        return []
    try:
        boot, start = stamp.rsplit(':', 1)
        start = int(start)
    except (ValueError, AttributeError):
        return []
    if boot != Path('/proc/sys/kernel/random/boot_id').read_text().strip():
        return []
    members = []
    for path in Path('/proc').iterdir():
        if not path.name.isdecimal():
            continue
        try:
            fields = (path/'stat').read_text().rsplit(')', 1)[1].split()
            if fields[0] != 'Z' and int(fields[2]) == pid and int(fields[3]) == pid and int(fields[19]) >= start:
                members.append(int(path.name))
        except (OSError, ValueError, IndexError):
            continue
    return members


def reap():
    # A missing observer does not prove the worker stopped. Never replay a task.
    with database() as db:
        for row in db.execute("SELECT * FROM jobs WHERE status IN ('starting','running','stopping')").fetchall():
            job = decode(row)
            if alive(job) or alive(job, 'child') or group_members(job):
                continue
            if job['status'] == 'starting' and time.time()-job['created_at'] < 30:
                continue
            job.update(status='cancelled' if job['status'] == 'stopping' else 'lost',
                       finished_at=time.time(), error='Supervisor and worker process are absent', notification='local')
            save(db, job)


def quota_module():
    spec = importlib.util.spec_from_file_location('agent_quota', SCRIPT.with_name('agent-quota.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def choose_home(parent_home, requested=None, allow_same=False):
    q = quota_module()
    homes = [home_named(requested)] if requested else registry()
    if not allow_same:
        homes = [h for h in homes if Path(h['path']).resolve() != Path(parent_home['path']).resolve()]
    if not homes:
        raise WorkerError('No other home selected; --allow-same-home explicitly permits the parent home')
    args = argparse.Namespace(refresh=False, timeout=12)
    # Consecutive spawns share the five-minute on-disk observations, including
    # pinned homes. A cache read never extends the observation's lifetime.
    rows = [q.fresh_codex_cache(h) for h in homes]
    if not any(row is not None for row in rows):
        # Populate the candidate pool once. Per-home locks merge simultaneous
        # spawns/CLI calls, and cached errors also suppress repeated requests.
        rows = q.query_many(homes, 'codex', args)
    candidates = [(h, r) for h, r in zip(homes, rows) if r is not None and q.score(r) is not None]
    if not candidates:
        details = '; '.join(r['home']+': '+r['status'] for r in rows if r is not None)
        retry_after = max((r.get('quota_retry_after_seconds', 0) for r in rows if r is not None), default=0)
        if retry_after:
            details += '; retry_after_seconds='+str(int(retry_after))
        raise WorkerError('No home with fresh usable quota. Run codex-quota --refresh. '+details)
    with database() as db:
        counts = {}
        for row in db.execute("SELECT data FROM jobs WHERE status IN ('starting','running','stopping')"):
            name = json.loads(row['data'])['home']['name']
            counts[name] = counts.get(name, 0)+1
    # Percentages are not absolute token balances. Rank the tightest returned window;
    # use active jobs only to break ties, never downgrade a requested model.
    return max(candidates, key=lambda hr: (q.score(hr[1]), -counts.get(hr[0]['name'], 0)))


def parent_id(args):
    value = args.parent or os.environ.get('CODEX_THREAD_ID')
    if not value:
        raise WorkerError('Use --parent SESSION_UUID outside a Codex session')
    try:
        return str(uuid.UUID(value))
    except ValueError:
        raise WorkerError('--parent must be a session UUID') from None


def worker_prompt(args, ident, task):
    writes = ', '.join(str(Path(p).resolve()) for p in args.write) or 'None; read-only task'
    npu = ('NPU is forbidden, including imports which initialize it.' if not args.npu_owner else
           f'NPU permitted only through npu-guard run --owner {args.npu_owner!r} --wait {args.npu_wait} '
           f'--timeout {args.npu_timeout} -- COMMAND. Keep device commands in foreground; no nested reservations.')
    return (f'You are cross-home worker {ident}. Follow applicable AGENTS.md.\n'
            f'Working directory: {args.cwd}\nAllowed writes: {writes}\n{npu}\n'
            'Do not modify other tasks or their memory; no memory Fork/Merge. Do not spawn additional '
            'agents unless the assigned task explicitly authorizes delegation.\n'
            'Finish with findings, changed files, checks and remaining issues. The supervisor saves your '
            'final answer and notifies the parent automatically; do not notify or queue the parent yourself.\n\n'
            'Assigned task:\n'+task+'\n')


def configured_spawn_admission(parent, writes, task_sha256=None, candidate_model=None, candidate_effort=None):
    """Called inside the existing BEGIN IMMEDIATE, just before slot insertion.

    The same registry transaction covers manual and automated spawns. Optional
    per-parent policy adds inherited capacity and cross-group write observation;
    no new reservation database or model approval is introduced.
    """
    path = ROOT / 'admission' / (parent + '.json')
    if not path.exists():
        return None
    try:
        with path.open('rb') as stream:
            body = stream.read(1024 * 1024 + 1)
        if len(body) > 1024 * 1024:
            raise ValueError('admission configuration too large')
        policy = json.loads(body)
        request = policy['request']
        if request['root_thread'] != parent:
            raise ValueError('admission parent binding mismatch')
        if Path(request['workers_db']).resolve() != (ROOT / 'workers.sqlite').resolve():
            raise ValueError('admission worker registry binding mismatch')
        helper = Path(policy['helper']['path'])
        if not helper.is_absolute() or hashlib.sha256(helper.read_bytes()).hexdigest() != policy['helper']['sha256']:
            raise ValueError('admission helper pin mismatch')
        request = dict(request, candidate_write_paths=[str(Path(p).resolve()) for p in writes], candidate_task_sha256=task_sha256,
                       candidate_model=candidate_model, candidate_effort=candidate_effort)
        checked = subprocess.run([sys.executable, str(helper), '--request', '-'],
                                 input=json.dumps(request), capture_output=True, text=True, timeout=5)
        result = json.loads(checked.stdout)
        if checked.returncode != 0 or result.get('admissible') is not True:
            raise WorkerError('Regional spawn admission deferred: ' + json.dumps(result, ensure_ascii=False)[:3000])
        return {'configuration': str(path), 'configuration_sha256': hashlib.sha256(body).hexdigest(),
                'observation': result}
    except WorkerError:
        raise
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
        raise WorkerError('Regional spawn admission unavailable: ' + str(exc)) from exc


def spawn(args):
    parent = parent_id(args)
    worker_peer_mail.hint(ROOT, parent)
    parent_home = home_named(args.parent_home or os.environ.get('CODEX_HOME', str(BASE/'.codex')))
    args.cwd = str(Path(args.cwd).resolve())
    if not Path(args.cwd).is_dir():
        raise WorkerError('--cwd must be an existing directory')
    if bool(args.npu_owner) != bool(args.npu_timeout):
        raise WorkerError('Specify --npu-owner and --npu-timeout together')
    if args.timeout <= 0 or args.npu_wait <= 0 or (args.npu_timeout is not None and args.npu_timeout <= 0):
        raise WorkerError('Timeouts must be positive')
    task = Path(args.task_file).read_text() if args.task_file else args.task
    if not task or not task.strip():
        raise WorkerError('Provide --task or --task-file')
    reap()
    home, quota = choose_home(parent_home, args.home, args.allow_same_home)
    config = tomllib.loads((Path(parent_home['path'])/'config.toml').read_text())
    ident = uuid.uuid4().hex[:16]
    directory = ROOT/'jobs'/ident
    prompt = worker_prompt(args, ident, task)
    job = dict(id=ident, parent=parent, status='starting', home=home, parent_home=parent_home,
               cwd=args.cwd, created_at=time.time(), timeout=args.timeout, notify=args.notify,
               notification='pending', directory=str(directory), quota=quota,
               model=args.model or config.get('model'), effort=args.effort or config.get('model_reasoning_effort'),
               write=args.write, sandbox=args.sandbox or ('workspace-write' if args.write else 'read-only'))
    with database() as db:
        count = db.execute("SELECT COUNT(*) FROM jobs WHERE parent=? AND status IN ('starting','running','stopping')", (parent,)).fetchone()[0]
        if count >= LIMIT:
            raise WorkerError(f'This parent already has {LIMIT} active workers; wait or cancel one')
        admission = configured_spawn_admission(parent, args.write, hashlib.sha256(task.encode()).hexdigest(), job['model'], job['effort'])
        if admission is not None:
            job['spawn_admission'] = admission
        directory.mkdir(parents=True, mode=0o700)
        (directory/'prompt.txt').write_text(prompt)
        db.execute('INSERT INTO jobs(id,parent,status,data) VALUES(?,?,?,?)',
                   (ident, parent, 'starting', json.dumps(job, ensure_ascii=False)))
        try:
            with (directory/'supervisor.log').open('ab') as log:
                child = subprocess.Popen([sys.executable, str(SCRIPT), '_run', ident],
                                         stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                         start_new_session=True, close_fds=True)
            job.update(supervisor_pid=child.pid, supervisor_stamp=process_stamp(child.pid))
            save(db, job)
        except OSError as error:
            job.update(status='failed', error=type(error).__name__, finished_at=time.time(), notification='local')
            save(db, job)
    return job


def kill_child(job, sig=signal.SIGTERM):
    if alive(job, 'child') or group_members(job):
        try:
            os.killpg(job['child_pid'], sig)
        except ProcessLookupError:
            pass


def stop_child(job, process=None):
    deadline = time.monotonic()+5
    kill_child(job)
    while (alive(job, 'child') or group_members(job)) and time.monotonic() < deadline:
        if process:
            process.poll()
        time.sleep(.1)
    kill_child(job, signal.SIGKILL)
    if process:
        process.wait(timeout=5)


def notify(ident):
    job = get_job(ident)
    # Serialize an automatic delivery and a simultaneous manual retry.
    with (Path(job['directory'])/'notification.lock').open('a') as lock:
        deadline = time.monotonic()+25
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    return get_job(ident)
                time.sleep(.1)
        return deliver(ident)


def deliver(ident):
    job = get_job(ident)
    if job.get('notification') == 'queued' or job['status'] in ACTIVE:
        return job
    route = None
    try:
        routes_path = ROOT/'notification_routes.json'
        rules = json.loads(routes_path.read_text()) if routes_path.exists() else {}
        rule = rules.get('parents', {}).get(job['parent'])
        if rule:
            spec = importlib.util.spec_from_file_location('_worker_completion_route', rule['resolver'])
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            route = module.resolve_completion(job)
            if route is None:
                return update(ident, notification='local', notification_route_pending=True,
                              notification_route_error='Delegated destination unavailable; original parent fallback forbidden')
    except (OSError, ValueError, KeyError, RuntimeError, ImportError) as error:
        return update(ident, notification='local', notification_route_pending=True,
                      notification_route_error=str(error)[:500])
    if job['notify'] == 'local' and route is None:
        return update(ident, notification='local')
    message = (f'[codex-worker {ident}] status={job["status"]}, home={job["home"]["name"]}. '
               f'Read {job["directory"]}/result.txt and inspect checks/logs before accepting the result. '
               f'Local inbox: codex-worker inbox --parent {job["parent"]}. '
               'This is an automatic completion notice, not a new user instruction.')
    # Local completion is durable before attempting delivery. Failure never loses results.
    try:
        with (Path(job['directory'])/'notification.log').open('ab') as log:
            done = subprocess.run([str(BINARY), 'queue', '--thread', route['thread'] if route else job['parent'], '--message', message],
                                  env=environment(route['home'] if route else job['parent_home']), stdin=subprocess.DEVNULL,
                                  stdout=log, stderr=log, timeout=20)
        return update(ident, notification='queued' if done.returncode == 0 else 'local',
                      notification_attempt_at=time.time(), notification_exit=done.returncode,
                      notification_target=route['thread'] if route else job['parent'],
                      notification_route_pending=done.returncode != 0)
    except (OSError, subprocess.TimeoutExpired) as error:
        return update(ident, notification='local', notification_attempt_at=time.time(), notification_error=type(error).__name__)


def run(ident):
    job = get_job(ident)
    directory = Path(job['directory'])
    stopped = []
    signal.signal(signal.SIGTERM, lambda *_: stopped.append(True))
    signal.signal(signal.SIGINT, lambda *_: stopped.append(True))
    if get_job(ident)['status'] == 'stopping':
        update(ident, status='cancelled', finished_at=time.time())
        notify(ident)
        return
    process = None
    try:
        cmd = [str(BINARY), '--no-daemon', '-a', 'never', 'exec', '--skip-git-repo-check',
               '--json', '--color', 'never', '-C', job['cwd'], '-s', job['sandbox'],
               '-o', str(directory/'result.txt')]
        if job.get('model'):
            cmd += ['-m', job['model']]
        if job.get('effort'):
            cmd += ['-c', 'model_reasoning_effort='+json.dumps(job['effort'])]
        for path in job['write']:
            p = Path(path).resolve()
            cmd += ['--add-dir', str(p if p.is_dir() else p.parent)]
        cmd += ['-']
        with (directory/'prompt.txt').open('rb') as prompt, (directory/'events.jsonl').open('wb') as events, (directory/'stderr.log').open('wb') as errors:
            process = subprocess.Popen(cmd, stdin=prompt, stdout=events, stderr=errors,
                                       env=environment(job['home']), start_new_session=True)
            # Retain the handle locally before persistence: DB failure must not orphan it.
            job.update(child_pid=process.pid, child_stamp=process_stamp(process.pid))
            job = update(ident, status='running', child_pid=job['child_pid'], child_stamp=job['child_stamp'])
            deadline = time.monotonic()+job['timeout']
            reason = None
            while process.poll() is None:
                if stopped or get_job(ident)['status'] == 'stopping':
                    reason = 'cancelled'
                    break
                if time.monotonic() >= deadline:
                    reason = 'timed_out'
                    break
                time.sleep(.3)
            if reason:
                stop_child(job, process)
            elif group_members(job):
                # A headless task is not complete with unowned background children.
                stop_child(job, process)
        # Exit 0 alone is insufficient: require completed turn and a final answer.
        completed = False
        thread_id = None
        with (directory/'events.jsonl').open(errors='replace') as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get('type') == 'thread.started':
                    thread_id = event.get('thread_id')
                if event.get('type') == 'turn.completed':
                    completed = True
                if event.get('type') == 'turn.failed':
                    completed = False
        result = directory/'result.txt'
        status = reason or ('completed' if process.returncode == 0 and completed and result.exists() and result.stat().st_size else 'failed')
        update(ident, status=status, finished_at=time.time(), exit_code=process.returncode, thread_id=thread_id)
    except BaseException as error:
        if process:
            try:
                stop_child(job, process)
            except subprocess.TimeoutExpired:
                pass
        still_live = alive(job, 'child') or bool(group_members(job))
        update(ident, status='stopping' if still_live else 'failed', finished_at=time.time(),
               child_pid=job.get('child_pid'), child_stamp=job.get('child_stamp'),
               error=type(error).__name__+': '+str(error))
    notify(ident)


def cancel(ident):
    with database() as db:
        row = db.execute('SELECT * FROM jobs WHERE id=?', (ident,)).fetchone()
        if row is None:
            raise WorkerError('Unknown job: '+ident)
        job = decode(row)
        if job['status'] not in ACTIVE:
            return job
        job['status'] = 'stopping'
        save(db, job)
    if alive(job):
        try:
            os.kill(job['supervisor_pid'], signal.SIGTERM)
        except ProcessLookupError:
            pass
    else:
        stop_child(job)
        reap()
    return job


def list_jobs(parent, inbox=False, ack=False):
    reap()
    with database() as db:
        rows = db.execute('SELECT * FROM jobs WHERE parent=? ORDER BY rowid', (parent,)).fetchall()
        jobs = [decode(row) for row in rows]
        if inbox:
            jobs = [j for j in jobs if j['status'] not in ACTIVE and not j['ack']]
            if ack:
                for job in jobs:
                    db.execute('UPDATE jobs SET ack=1 WHERE id=?', (job['id'],))
    return jobs


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description='Dispatch up to 8 Codex workers per parent across homes, with quota selection and completion inbox.')
    commands = parser.add_subparsers(dest='command', required=True)
    spawn_p = commands.add_parser('spawn', help='Start one supervised worker; default selects another home by quota')
    spawn_p.add_argument('--parent')
    spawn_p.add_argument('--parent-home')
    spawn_p.add_argument('--home', help='Pin a home; default auto-selects by fresh quota')
    spawn_p.add_argument('--allow-same-home', action='store_true')
    spawn_p.add_argument('--cwd', default=os.getcwd())
    tasks = spawn_p.add_mutually_exclusive_group(required=True)
    tasks.add_argument('--task')
    tasks.add_argument('--task-file')
    spawn_p.add_argument('--write', action='append', default=[], help='Allowed output file/directory; repeat as needed')
    spawn_p.add_argument('--model')
    spawn_p.add_argument('--effort', choices=['low', 'medium', 'high', 'xhigh', 'max', 'ultra'])
    spawn_p.add_argument('--timeout', type=float, default=1800, help='Worker runtime in seconds')
    spawn_p.add_argument('--notify', choices=['queue', 'local'], default='queue')
    spawn_p.add_argument('--sandbox', choices=['read-only', 'workspace-write', 'danger-full-access'])
    spawn_p.add_argument('--npu-owner')
    spawn_p.add_argument('--npu-wait', type=int, default=600, help='Maximum queue wait in seconds; separate from device execution timeout')
    spawn_p.add_argument('--npu-timeout', type=int)
    peer_p = commands.add_parser('peer-inbox', help='Read durable peer messages; independent of worker completion inbox')
    peer_p.add_argument('--parent')
    peer_p.add_argument('--ack', action='append', default=[], metavar='MESSAGE_ID')
    peer_p.add_argument('--limit', type=int, default=10)
    route_p = commands.add_parser('suppress-parent-notifications', help='Make existing live supervisors keep completions local for configured delegated routing; ownership unchanged')
    route_p.add_argument('--parent', required=True)
    for name in ['status', 'result', 'cancel', 'notify', '_run']:
        p = commands.add_parser(name)
        p.add_argument('job')
    for name in ['list', 'inbox', 'watch']:
        p = commands.add_parser(name)
        p.add_argument('--parent')
        if name == 'inbox':
            p.add_argument('--ack', action='store_true', help='Acknowledge returned messages')
        if name == 'watch':
            p.add_argument('--timeout', type=float, default=60, help='Wait up to this many seconds for completion')
    args = parser.parse_args()
    if args.command == '_run':
        run(args.job)
        return
    if args.command == 'spawn':
        value = spawn(args)
    elif args.command == 'suppress-parent-notifications':
        parent = parent_id(args)
        rules = json.loads((ROOT/'notification_routes.json').read_text())
        if parent not in rules.get('parents', {}):
            raise WorkerError('Configure a durable completion resolver before suppressing parent delivery')
        changed = []
        with database() as db:
            for row in db.execute('SELECT * FROM jobs WHERE parent=?', (parent,)).fetchall():
                job = decode(row)
                if job['status'] not in ACTIVE:
                    continue
                job['notify'] = 'local'
                job['notification_route_pending'] = True
                save(db, job)
                changed.append(job['id'])
        value = {'parent': parent, 'changed': changed, 'ownership_unchanged': True}
    elif args.command == 'peer-inbox':
        parent = parent_id(args)
        receipts = worker_peer_mail.acknowledge(ROOT, parent, args.ack)
        value = {'messages': worker_peer_mail.read(ROOT, parent, max(1, min(args.limit, 32))), 'acknowledgements': receipts}
    elif args.command in ('list', 'inbox', 'watch'):
        parent = parent_id(args)
        deadline = time.monotonic()+getattr(args, 'timeout', 0)
        while True:
            value = list_jobs(parent, args.command != 'list', getattr(args, 'ack', False))
            if value or args.command != 'watch' or time.monotonic() >= deadline:
                break
            time.sleep(.5)
        worker_peer_mail.hint(ROOT, parent)
    elif args.command == 'cancel':
        value = cancel(args.job)
    elif args.command == 'notify':
        value = notify(args.job)
    else:
        reap()
        value = get_job(args.job)
        # Status/result callers may never use list/inbox. Surface the caller's
        # own durable maintenance mail without changing worker JSON on stdout.
        caller = os.environ.get('CODEX_THREAD_ID')
        if caller:
            try:
                worker_peer_mail.hint(ROOT, str(uuid.UUID(caller)))
            except (OSError, ValueError, KeyError, TypeError) as error:
                print('[codex-worker peer mail] Optional hint unavailable (' +
                      type(error).__name__ + ').', file=sys.stderr)
        if args.command == 'result':
            path = Path(value['directory'])/'result.txt'
            if not path.exists():
                raise WorkerError('No final answer yet; status='+value['status']+'; logs='+value['directory'])
            print(path.read_text())
            return
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (WorkerError, OSError, ValueError, sqlite3.Error) as error:
        print('codex-worker: '+str(error), file=sys.stderr)
        raise SystemExit(2)
