"""Unmodified canonical checker, private provider injection and payload evidence."""
import os, sys, json, hashlib, time, tempfile
from dataclasses import replace
from pathlib import Path
BASE = Path(__file__).resolve().parent
CASE_ID = sys.argv[1]
assert CASE_ID in ('throughput-4k-complex128', 'scientific-1m-complex128')
P = BASE / CASE_ID
W = Path('/root/dyq/workspace')
T = W / 'TileOPs'
m = json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text())
assert m['phase'] == 'running' and m['child']['pid'] == os.getpgrp()
assert m['owner'] == 'baseline700-20261026/fft_remaining_c128_2'
(P / 'GUARD_METADATA.json').write_text(json.dumps(m, indent=2))
os.chdir(P)
short = Path(f"/proc/{os.getpid()}/cwd")
sys.dont_write_bytecode = True
for n in ('tmp', 'ASCEND_CACHE_PATH', 'ASCEND_WORK_PATH', 'XDG_CACHE_HOME', 'TORCH_EXTENSIONS_DIR', 'ASCEND_PROCESS_LOG_PATH'):
    os.environ[n] = str(short / n)
os.environ['TMPDIR'] = str(short / 'tmp')
tempfile.tempdir = str(short / 'tmp')
os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = '0'
os.environ['BASELINE_PROGRESS_FILE'] = str(P / 'progress.json')
vendor = W / '.cache/ascend-classic/native-pilot-20261020/ops-math/install/vendors/classicpilot_math'
os.environ['ASCEND_CLASSIC_MATH_VENDOR'] = str(vendor)
os.environ['ASCEND_CUSTOM_OPP_PATH'] = str(vendor)
sys.path[:0] = [str(BASE), str(T), str(T / 'src'), str(W)]
import benchmarks.ascend_all_pairs.ops_math as provider
from adapter import make
calls = []

def recorded_make(case):
    import torch
    candidate = make(case)
    original = candidate.implementation.run
    import inspect
    window = inspect.getclosurevars(original).nonlocals['window']
    torch.save(window.cpu(), P / 'window.pt')
    (P / 'WINDOW_BINDING.json').write_text(json.dumps({'shape':list(window.shape), 'dtype':str(window.dtype), 'device':str(window.device), 'contiguous':window.is_contiguous(), 'data_ptr':window.data_ptr(), 'non_null':bool(window.data_ptr()), 'all_one_plus_zero_j':bool(torch.equal(window.cpu(), torch.ones(window.numel(),dtype=window.dtype))), 'payload':'window.pt'},indent=2))
    def recorded_run(*args):
        idx = len(calls)
        before = tuple(x.cpu().clone() for x in args)
        torch.save(before, P / f'input_{idx}_before.pt')
        event = {'index': idx, 'input_before': f'input_{idx}_before.pt', 'returned': False}
        calls.append(event)
        (P / 'PAYLOADS.json').write_text(json.dumps(calls, indent=2))
        begin = time.monotonic()
        actual = original(*args)
        torch.npu.synchronize()
        event['api_call_and_sync_wall_s'] = time.monotonic() - begin
        torch.save(actual.cpu(), P / f'output_{idx}.pt')
        after = tuple(x.cpu() for x in args)
        torch.save(after, P / f'input_{idx}_after.pt')
        event.update(returned=True, output=f'output_{idx}.pt', input_after=f'input_{idx}_after.pt',
                     full_output_elements=actual.numel(), input_preserved=all(torch.equal(a,b) for a,b in zip(before,after)))
        (P / 'PAYLOADS.json').write_text(json.dumps(calls, indent=2))
        return actual
    def recorded_cleanup():
        candidate.cleanup()
        (P / 'CLEANUP.json').write_text(json.dumps({'synchronized_and_cleanup': True}))
    return replace(candidate, implementation=replace(candidate.implementation, run=recorded_run), cleanup=recorded_cleanup)

provider.PROVIDERS['FFTC2CFwdOp'] = recorded_make
checker = T / 'scripts/ascend_pair_check.py'
snapshot = BASE / 'ascend_pair_check.snapshot.py'
assert snapshot.read_bytes() == checker.read_bytes(), 'Checker changed after CPU freeze'
sys.argv = [str(checker), '--op', 'FFTC2CFwdOp', '--case-id', CASE_ID, '--library', 'ops-math', '--output', str(P/'checker.json')]
rc = 1
try:
    exec(compile(snapshot.read_bytes(), str(checker), 'exec'), {'__name__': '__main__', '__file__': str(checker)})
except SystemExit as exc:
    rc = exc.code
finally:
    (P / 'CHILD_RESULT.json').write_text(json.dumps({'returncode':rc, 'checker_unchanged':snapshot.read_bytes()==checker.read_bytes()}, indent=2))
raise SystemExit(rc)
