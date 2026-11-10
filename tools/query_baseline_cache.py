#!/usr/bin/env python3
"""Read cached baseline timings without invoking a device or rerunning a pair.

Run on the same host/cohort as the intended comparison. Results always expose
their runtime identity; software/hardware/protocol changes require remeasurement.
"""
import argparse
import hashlib
import json
from pathlib import Path
import socket
from baseline_cache_guard_binding import validate_binding

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, required=True, help='host CACHE.jsonl')
    p.add_argument('--op', required=True)
    p.add_argument('--case-id', required=True)
    p.add_argument('--library')
    p.add_argument('--allow-foreign-host', action='store_true', help='inspect only; foreign timings are not comparable local cache hits')
    args = p.parse_args()
    hits, rejected = [], []
    native_memo_path = args.cache.parent / 'native_hashes.json'
    native_memo = json.loads(native_memo_path.read_text()) if native_memo_path.exists() else {}
    checked_native = {}
    binding_file = args.cache.parent / 'GUARD_BINDINGS.jsonl'
    bindings = {b['cache_key']: b for b in map(json.loads, binding_file.read_text().splitlines())} if binding_file.exists() else {}
    for line in args.cache.read_text().splitlines():
        r = json.loads(line)
        lib, op, case = r['key']
        if (op,case) != (args.op,args.case_id) or args.library and lib != args.library:
            continue
        reasons = []
        identity = r['identity']
        if not r.get('performance_valid') or r.get('status') != 'cached':
            reasons.append('invalid performance receipt')
        foreign = identity['environment']['hostname'] != socket.gethostname()
        binding = r.get('guard_binding') or bindings.get(r['cache_key'])
        reasons.extend(validate_binding(r, binding, check_implementation=not foreign))
        if binding:
            r['guard_binding'] = binding
        if foreign and not args.allow_foreign_host:
            reasons.append('foreign host cohort')
        if not foreign:
            for name, expected in identity['sources'].items():
                path = Path(name)
                if not path.is_absolute():
                    path = ROOT / path
                if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                    reasons.append('changed source/component: '+name)
            core = ROOT / 'tools/measure_baselines.py'
            if hashlib.sha256(core.read_bytes()).hexdigest() != identity['protocol']['core_sha256']:
                reasons.append('changed benchmark protocol implementation')
            for name, expected in identity['environment'].get('loaded_native_sha256', {}).items():
                path = Path(name)
                if not path.is_file():
                    reasons.append('missing native runtime: '+name)
                    continue
                if name not in checked_native:
                    s = path.stat()
                    stamp = [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]
                    memo = native_memo.get(name, {})
                    if memo.get('stamp') == stamp:
                        checked_native[name] = memo['sha256']
                    else:
                        h = hashlib.sha256()
                        with path.open('rb') as f:
                            for block in iter(lambda:f.read(8*1024*1024),b''):
                                h.update(block)
                        checked_native[name] = h.hexdigest()
                if checked_native[name] != expected:
                    reasons.append('changed native runtime: '+name)
        if reasons:
            rejected.append({'key':r['key'],'reasons':reasons})
        else:
            hits.append(r)
    print(json.dumps({'hits':hits, 'rejected':rejected,
        'reuse_rule':'Same full case contract, host/software cohort, source/native hashes and protocol; compare device and host-roundtrip metrics separately.'},ensure_ascii=False,indent=2))
    return 0 if hits else 1


if __name__ == '__main__':
    raise SystemExit(main())
