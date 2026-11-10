#!/usr/bin/env python3
"""CPU-only audit of a frozen cache prefix; never reruns a baseline or imports NPU."""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import socket
import statistics
import time

from baseline_cache_guard_binding import validate_binding

ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str,
                                    separators=(',', ':')).encode()).hexdigest()


class Audit:
    def __init__(self, root, hostdir, fresh=False):
        self.root, self.hostdir = Path(root), Path(hostdir)
        memo = self.hostdir / 'native_hashes.json'
        self.memo = {} if fresh or not memo.exists() else json.loads(memo.read_text())
        self.hashes, self.bytes_hashed = {}, 0
        binding = self.hostdir / 'GUARD_BINDINGS.jsonl'
        self.bindings = {r['cache_key']: r for r in map(json.loads, binding.read_text().splitlines())} if binding.exists() else {}
        self.tasks = {t['id']: t for t in json.loads((self.hostdir.parent / 'PLAN.json').read_text())['pairs']}

    def sha(self, name):
        p = Path(name)
        if not p.is_absolute():
            p = self.root / p
        name = str(p)
        if name in self.hashes:
            return self.hashes[name]
        if not p.is_file():
            self.hashes[name] = None
            return None
        before = p.stat()
        stamp = lambda s: [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]
        memo = self.memo.get(name, {})
        if memo.get('stamp') == stamp(before):
            value = memo['sha256']
        else:
            h = hashlib.sha256()
            with p.open('rb') as f:
                for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
                    h.update(block)
                    self.bytes_hashed += len(block)
            value = h.hexdigest()
        if stamp(before) != stamp(p.stat()):
            raise ValueError('file changed during audit: ' + name)
        self.hashes[name] = value
        return value

    def check(self, row):
        errors, notes = [], []
        def require(ok, reason):
            if not ok:
                errors.append(reason)
        i = row['identity']
        task = self.tasks.get(row['task_id'])
        require(task is not None, 'task absent from campaign PLAN')
        if task:
            require(task['key'] == row['key'] and task['case'] == i['case'], 'exact case/PLAN mismatch')
            require(task['host'] == self.hostdir.name, 'wrong assigned host')
        require(row['cache_key'] == digest(i), 'cache identity digest mismatch')
        require(row['status'] == 'cached' and row.get('performance_valid') is True, 'invalid cache status')
        require(i['environment']['hostname'] == socket.gethostname(), 'foreign host cohort')
        require(i['library'] == row['key'][0], 'library identity mismatch')
        c = json.loads(Path(row['correctness_receipt']).read_text())
        require(self.sha(row['correctness_receipt']) == row['correctness_sha256'], 'changed correctness receipt')
        require([c['library'], c['op'], c['case_id']] == row['key'], 'wrong checker exact key')
        v = c.get('verification', {})
        require(c.get('status') == 'verified' and v.get('checked_outputs', 0) > 0
                and v.get('checked_outputs') == v.get('total_outputs') and not v.get('unchecked_reason'), 'incomplete original output verification')
        require(c.get('inputs_preserved') is True and not c.get('provider_source_changed')
                and not c.get('oracle_source_changed'), 'input/source protection missing')
        snapshot = c['case_snapshot']
        require(snapshot['parameters'] == i['case']['params'] and snapshot['declared_tensors'] == i['case']['tensors'], 'checker case contract mismatch')
        g = json.loads(Path(row['guard_receipt']).read_text())
        require(g.get('owner') == 'baseline-cache-20261027/' + self.hostdir.name + '/' + row['task_id'], 'guard owner mismatch')
        require(g.get('phase') == 'finished' and g.get('performance_valid') is True
                and g.get('workload_exit_code') == 0 and not g.get('conflict_detected')
                and not g.get('visibility_lost'), 'guard does not certify exclusive successful run')
        binding = row.get('guard_binding') or self.bindings.get(row['cache_key'])
        errors.extend(validate_binding(row, binding))
        if binding and binding['evidence'] != 'captured_before_and_after_measurement':
            notes.append(binding['evidence'])
        protocol = i['protocol']
        require((protocol.get('id'), protocol.get('warmup'), protocol.get('repeats'), protocol.get('cache_flush_mib'))
                == ('ascend950-isolated-ffts-v2', 3, 15, 1024), 'wrong measurement protocol')
        require(self.sha('tools/measure_baselines.py') == protocol['core_sha256'], 'changed benchmark core')
        require(row['timing'].get('measurement_protocol') == protocol['id'], 'timing protocol mismatch')
        for samples, mean, label in ((row['timing']['latency_samples_ms'], row['timing']['latency_ms'], 'profiler'),
                                     (row['end_to_end_samples_ms'], row['end_to_end_mean_ms'], 'host roundtrip')):
            valid = len(samples) == 15 and all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0 for x in samples)
            require(valid, label + ' incomplete/nonpositive samples')
            if valid:
                require(math.isclose(mean, statistics.mean(samples), rel_tol=1e-10, abs_tol=1e-10), label + ' mean differs from all15 samples')
        metric = 'device_latency_ms' if row['timing_mode'] == 'device_async' else 'end_to_end_mean_ms'
        require(row['comparison_metric'] == metric, 'mixed timing metric')
        for name, expected in i['sources'].items():
            require(self.sha(name) == expected, 'changed source/component: ' + name)
        for name, expected in i['environment'].get('loaded_native_sha256', {}).items():
            require(self.sha(name) == expected, 'changed native runtime: ' + name)
        for field, file in [('verifier_sha256', 'TileOPs/scripts/cache_frozen_pair_check.py'),
                            ('adapter_sha256', 'TileOPs/benchmarks/ascend_baselines.py'),
                            ('registry_sha256', 'TileOPs/benchmarks/ascend_all_pairs/registry.py')]:
            require(self.sha(file) == i[field], 'changed ' + field)
        if self.sha('tools/measure_baseline_cache_pair.py') != i['wrapper_sha256']:
            notes.append('historical_measurement_wrapper_differs_current; core/protocol independently checked')
        flush = i.get('exported_flush')
        if flush:
            directory = self.hostdir.parent / 'flush'
            for name, field in [('host_kernel.asc', 'generated_source_sha256'), ('flush.aibin', 'device_binary_sha256'), ('flush.so', 'library_sha256')]:
                require(self.sha(directory / name) == flush[field], 'changed exported L2 flush: ' + name)
            require(flush.get('whole_buffer_check') == '268435456 int32 ones all overwritten with zero', 'missing whole1GiB flush check')
        else:
            notes.append('original_JIT_flush; unified_core_marker_self_test_required_by_core')
        return errors, notes


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', required=True, type=Path)
    p.add_argument('--limit', type=int, help='freeze and audit first N durable cache records')
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--fresh-hashes', action='store_true')
    a = p.parse_args()
    lines = a.cache.read_text().splitlines()
    lines = lines[:a.limit] if a.limit is not None else lines
    rows = [json.loads(line) for line in lines]
    if a.limit is not None and len(rows) != a.limit:
        raise ValueError('requested cache prefix not present')
    audit = Audit(ROOT, a.cache.parent, a.fresh_hashes)
    start = time.time()
    seen, result, note_counts = set(), [], collections.Counter()
    for row in rows:
        try:
            errors, notes = audit.check(row)
        except (KeyError, OSError, ValueError, TypeError) as exc:
            errors, notes = [type(exc).__name__ + ': ' + str(exc)], []
        if row['task_id'] in seen:
            errors.append('duplicate cache task_id')
        seen.add(row['task_id'])
        note_counts.update(notes)
        result.append({'task_id': row['task_id'], 'key': row['key'], 'cache_key': row['cache_key'],
                       'usable': not errors, 'errors': errors, 'notes': notes})
    report = {'schema': 1, 'started_at': start, 'checked_at': time.time(), 'host': a.cache.parent.name,
              'cache': str(a.cache.resolve()), 'snapshot_prefix_sha256': hashlib.sha256(('\n'.join(lines)+'\n').encode()).hexdigest(),
              'checked': len(rows), 'usable': sum(r['usable'] for r in result),
              'rejected': sum(not r['usable'] for r in result), 'notes': dict(note_counts),
              'files_hashed_or_stamp_checked': len(audit.hashes), 'bytes_freshly_hashed': audit.bytes_hashed,
              'fresh_hashes': a.fresh_hashes, 'device_runs': 0, 'rows': result,
              'reuse_boundary': 'Same exact case, host, baseline/reference source, native runtime and measurement protocol; historical successful execution evidence, no new device execution.'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as f:
        f.write(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, ensure_ascii=False))
    return 0 if not report['rejected'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
