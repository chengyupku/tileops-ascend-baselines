"""CPU-only regression tests; no credentials, inference or hardware access."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import textwrap
import time
import unittest
from unittest.mock import patch
import uuid

spec = importlib.util.spec_from_file_location('worker', Path(__file__).with_name('codex-worker.py'))
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.patch = patch.object(w, 'ROOT', self.root/'workers')
        self.patch.start()
        self.signals = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        self.home = {'name': 'codex-0', 'path': str(self.root/'home')}
        Path(self.home['path']).mkdir()
        (Path(self.home['path'])/'config.toml').write_text('model="test-model"\nmodel_reasoning_effort="high"\n')
        self.parent = str(uuid.uuid4())

    def tearDown(self):
        for sig, handler in self.signals.items():
            signal.signal(sig, handler)
        self.patch.stop()
        self.temp.cleanup()

    def args(self, **changes):
        data = dict(parent=self.parent, parent_home='codex-0', cwd=str(self.root),
                    npu_owner=None, npu_wait=600, npu_timeout=None, timeout=1, task='CPU test', task_file=None,
                    home=None, allow_same_home=False, write=[], notify='queue', model=None, effort=None, sandbox=None)
        data.update(changes)
        return argparse.Namespace(**data)

    def seed(self, status='starting', task='complete', **changes):
        ident = uuid.uuid4().hex[:16]
        directory = w.ROOT/'jobs'/ident
        directory.mkdir(parents=True)
        (directory/'prompt.txt').write_text(task)
        job = dict(id=ident, parent=self.parent, home=self.home, parent_home=self.home,
                   status=status, created_at=time.time()-60, directory=str(directory), cwd=str(self.root),
                   timeout=.4, notify='queue', notification='pending', write=[], sandbox='read-only')
        job.update(changes)
        with w.database() as db:
            db.execute('INSERT INTO jobs(id,parent,status,data) VALUES(?,?,?,?)',
                       (ident, self.parent, status, json.dumps(job)))
        return job

    def fake_binary(self):
        p = self.root/'fake-codex'
        p.write_text(textwrap.dedent('''\
            #!/usr/bin/env python3
            import json,sys,time
            import os,signal
            from pathlib import Path
            if 'queue' in sys.argv:
                print('queued')
                sys.exit(1 if Path(__file__).with_name('queue-fail').exists() else 0)
            prompt=sys.stdin.read()
            if prompt=='hang': time.sleep(60)
            if prompt=='stubborn-descendant':
                child=os.fork()
                if child==0:
                    signal.signal(signal.SIGTERM,signal.SIG_IGN)
                    time.sleep(60)
                    sys.exit(0)
                Path(__file__).with_name('descendant-pid').write_text(str(child))
                time.sleep(60)
            if 'wait-for-test-gate' in prompt:
                while not Path(__file__).with_name('gate').exists(): time.sleep(.05)
            print(json.dumps({'type':'thread.started','thread_id':'test-thread'}),flush=True)
            if prompt=='false-success': sys.exit(0)
            if prompt=='failed':
                print(json.dumps({'type':'turn.failed'}),flush=True)
                sys.exit(1)
            output=Path(sys.argv[sys.argv.index('-o')+1])
            output.write_text('TEST RESULT')
            print(json.dumps({'type':'turn.completed'}),flush=True)
            '''))
        p.chmod(0o700)
        return p

    def test_consecutive_and_pinned_dispatches_reuse_quota_until_300_seconds(self):
        q = w.quota_module()
        homes = []
        for i in range(3):
            root = self.root/('candidate-'+str(i)); root.mkdir(); (root/'auth.json').write_text('{}')
            homes.append({'name': 'codex-'+str(i+1), 'path': str(root)})
        def answer(home, _):
            return q.codex_bucket(home, {'primary': {'usedPercent': 10 if home == homes[1] else 50}}, 'pro')
        with patch.object(w, 'quota_module', return_value=q), patch.object(w, 'registry', return_value=homes), patch.object(q, 'CACHE', self.root/'quota'), patch.object(q.time, 'time', return_value=1000) as clock, patch.object(q, 'codex_query', side_effect=answer) as query:
            self.assertEqual(w.choose_home(self.home)[0], homes[1])
            self.assertEqual(query.call_count, 3)
            for _ in range(8):
                self.assertEqual(w.choose_home(self.home)[0], homes[1])
            clock.return_value = 1299
            self.assertEqual(w.choose_home(self.home, requested='codex-1')[0], homes[0])
            self.assertEqual(query.call_count, 3)
            clock.return_value = 1300
            self.assertEqual(w.choose_home(self.home)[0], homes[1])
            self.assertEqual(query.call_count, 6)

    def test_eight_workers_per_parent_atomic_under_concurrent_spawns(self):
        class Process:
            pid = os.getpid()
        with patch.object(w, 'home_named', return_value=self.home), \
             patch.object(w, 'choose_home', return_value=(self.home, {})), \
             patch.object(w.subprocess, 'Popen', return_value=Process()):
            # Exercise 8 separate parents (each may be a different home).
            for _ in range(8):
                self.parent = str(uuid.uuid4())
                def launch(_):
                    try:
                        return w.spawn(self.args())
                    except w.WorkerError:
                        return None
                with ThreadPoolExecutor(max_workers=12) as pool:
                    jobs = list(pool.map(launch, range(12)))
                self.assertEqual(sum(job is not None for job in jobs), 8)
                self.assertEqual(len(w.list_jobs(self.parent)), 8)

    def test_success_persists_result_and_delivers_once(self):
        job = self.seed()
        with patch.object(w, 'BINARY', self.fake_binary()):
            w.run(job['id'])
            completed = w.get_job(job['id'])
            self.assertEqual(completed['status'], 'completed')
            self.assertEqual(completed['notification'], 'queued')
            self.assertEqual((Path(job['directory'])/'result.txt').read_text(), 'TEST RESULT')
            with patch.object(w.subprocess, 'run', side_effect=AssertionError('duplicate notification')):
                w.notify(job['id'])
        self.assertEqual(len(w.list_jobs(self.parent, inbox=True)), 1)
        self.assertEqual(len(w.list_jobs(self.parent, inbox=True)), 1)
        self.assertEqual(len(w.list_jobs(self.parent, inbox=True, ack=True)), 1)
        self.assertEqual(w.list_jobs(self.parent, inbox=True), [])

    def test_eight_real_supervisors_release_slots_after_cancel(self):
        # Full subprocess lifecycle with a local stand-in for the inference binary.
        runner = self.root/'runner.py'
        binary = self.fake_binary()
        runner.write_text('import importlib.util\nfrom pathlib import Path\n'
                          f's=importlib.util.spec_from_file_location("w", {str(Path(w.__file__))!r})\n'
                          'w=importlib.util.module_from_spec(s);s.loader.exec_module(w)\n'
                          f'w.ROOT=Path({str(w.ROOT)!r});w.BINARY=Path({str(binary)!r})\n'
                          'w.main()\n')
        children = []
        popen = subprocess.Popen
        def track(*args, **kwargs):
            p = popen(*args, **kwargs)
            children.append(p)
            return p
        jobs = []
        try:
            with patch.object(w, 'SCRIPT', runner), patch.object(w, 'home_named', return_value=self.home), \
                 patch.object(w, 'choose_home', return_value=(self.home, {})), patch.object(w.subprocess, 'Popen', side_effect=track):
                for _ in range(8):
                    jobs.append(w.spawn(self.args(task='wait-for-test-gate', notify='local', timeout=30)))
                with self.assertRaises(w.WorkerError):
                    w.spawn(self.args())
                deadline = time.monotonic()+15
                while time.monotonic() < deadline:
                    states = [w.get_job(j['id']) for j in jobs]
                    if all(j['status']=='running' and w.alive(j,'child') for j in states):
                        break
                    time.sleep(.1)
                self.assertTrue(all(j['status']=='running' and w.alive(j,'child') for j in states))
                w.cancel(jobs[0]['id'])
                children[0].wait(timeout=10)
                self.assertEqual(w.get_job(jobs[0]['id'])['status'], 'cancelled')
                jobs.append(w.spawn(self.args(task='wait-for-test-gate', notify='local', timeout=30)))
                (self.root/'gate').touch()
            for p in children:
                p.wait(timeout=10)
            self.assertTrue(all(w.get_job(j['id'])['status']=='completed' for j in jobs[1:]))
            self.assertEqual(len(w.list_jobs(self.parent, inbox=True)), 9)
        finally:
            (self.root/'gate').touch()
            for job in jobs:
                w.cancel(job['id'])
            for p in children:
                p.wait(timeout=10)

    def test_failed_delivery_stays_in_inbox_and_can_retry(self):
        job = self.seed()
        flag = self.root/'queue-fail'
        flag.touch()
        with patch.object(w, 'BINARY', self.fake_binary()):
            w.run(job['id'])
            self.assertEqual(w.get_job(job['id'])['notification'], 'local')
            self.assertEqual(len(w.list_jobs(self.parent, inbox=True)), 1)
            flag.unlink()
            self.assertEqual(w.notify(job['id'])['notification'], 'queued')

    def test_concurrent_notification_retries_deliver_only_once(self):
        job = self.seed(status='completed')
        with patch.object(w, 'BINARY', self.fake_binary()), patch.object(w.subprocess, 'run', wraps=subprocess.run) as delivery:
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(lambda _: w.notify(job['id']), range(8)))
            self.assertEqual(delivery.call_count, 1)

    def test_child_cleanup_survives_database_write_failure(self):
        job = self.seed(task='hang', notify='local')
        original = w.update
        child = {}
        def fail_on_running(ident, **changes):
            if changes.get('status')=='running':
                child.update(changes)
                raise w.sqlite3.OperationalError('simulated persistence failure')
            return original(ident, **changes)
        with patch.object(w, 'BINARY', self.fake_binary()), patch.object(w, 'update', side_effect=fail_on_running):
            w.run(job['id'])
        self.assertEqual(w.get_job(job['id'])['status'], 'failed')
        self.assertFalse(w.alive(child, 'child'))

    def test_cancel_kills_descendant_after_leader_exits(self):
        job = self.seed(task='stubborn-descendant', notify='local')
        with patch.object(w, 'BINARY', self.fake_binary()):
            w.run(job['id'])
        finished = w.get_job(job['id'])
        self.assertEqual(finished['status'], 'timed_out')
        self.assertEqual(w.group_members(finished), [])
        self.assertIsNone(w.process_stamp(int((self.root/'descendant-pid').read_text())))

    def test_fail_timeout_and_exit_zero_without_result_are_not_success(self):
        with patch.object(w, 'BINARY', self.fake_binary()):
            for task, expected in [('failed','failed'), ('false-success','failed'), ('hang','timed_out')]:
                job = self.seed(task=task)
                w.run(job['id'])
                done = w.get_job(job['id'])
                self.assertEqual(done['status'], expected)
                self.assertFalse(w.alive(done, 'child'))

    def test_cancel_before_launch_never_runs_inference(self):
        job = self.seed(status='stopping', notify='local')
        with patch.object(w.subprocess, 'Popen', side_effect=AssertionError('must not launch')):
            w.run(job['id'])
        self.assertEqual(w.get_job(job['id'])['status'], 'cancelled')

    def test_dead_supervisor_does_not_free_slot_while_child_is_live(self):
        child = subprocess.Popen(['sleep','30'])
        try:
            job = self.seed(status='running', child_pid=child.pid, child_stamp=w.process_stamp(child.pid))
            w.reap()
            self.assertEqual(w.get_job(job['id'])['status'], 'running')
        finally:
            child.terminate()
            child.wait()
        w.reap()
        self.assertEqual(w.get_job(job['id'])['status'], 'lost')
        self.assertEqual(len(w.list_jobs(self.parent, inbox=True)), 1)

    def test_pid_reuse_cannot_signal_unrelated_process(self):
        job = self.seed(status='running', child_pid=os.getpid(), child_stamp='wrong-boot-and-start')
        with patch.object(os, 'killpg', side_effect=AssertionError('PID was reused')):
            w.kill_child(job)

    def test_environment_does_not_borrow_parent_credentials_or_thread(self):
        with patch.dict(os.environ, {'OPENAI_API_KEY':'fake','CODEX_THREAD_ID':'parent','CODEX_HOME':'parent'}):
            env = w.environment(self.home)
        self.assertEqual(env['CODEX_HOME'], self.home['path'])
        self.assertNotIn('OPENAI_API_KEY', env)
        self.assertNotIn('CODEX_THREAD_ID', env)

    def test_selection_ignores_parent_stale_and_exhausted(self):
        q = w.quota_module()
        homes = [dict(name=f'codex-{i}', path=str(self.root/f'home{i}')) for i in range(4)]
        rows = {h['name']:q.codex_bucket(h, {'primary':{'usedPercent':used}}, 'pro')
                for h,used in zip(homes, [1,10,30,100])}
        rows['codex-1']['stale'] = True
        with patch.object(w, 'registry', return_value=homes), patch.object(w, 'quota_module', return_value=q), \
             patch.object(q, 'query', side_effect=lambda home,*_:rows[home['name']]):
            home, row = w.choose_home(homes[0])
            self.assertEqual(home['name'], 'codex-2')
            rows['codex-2']['stale'] = True
            with self.assertRaises(w.WorkerError):
                w.choose_home(homes[0])


if __name__ == '__main__':
    unittest.main()
