"""Adopt only verified public configurations and the exhaustively checked BF16 law."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
KERNELS = ROOT / 'TileOPs/src/tileops/kernels/ascend/elementwise'
candidate = HERE / 'candidate/_logroots_v10.py'
candidate_hash = hashlib.sha256(candidate.read_bytes()).hexdigest()
assert candidate_hash == '68a766c046c69c6e4d4119467830375a0ab86396ca06353fc8e62ea0342fdbd1'


def receipt(label):
    rows = [json.loads(p.read_text()) for p in (HERE / 'requests').glob(label + '_*/status.json')]
    result = max(rows, key=lambda r: r['completed_at'])
    assert result['guard_exit_code'] == 0, result
    return result


proof_path = HERE / 'search/source-equivalence-logbf16-v10.json'
proof = json.loads(proof_path.read_text())
receipt(proof['guard_label'])
assert proof['new_helper_sha256'] == candidate_hash
assert proof['old_helper_sha256'] == hashlib.sha256((KERNELS / '_logroots.py').read_bytes()).hexdigest()
assert len(proof['comparisons']) == 18 and all(r['exact_match'] for r in proof['comparisons'])
for r in proof['comparisons']:
    assert hashlib.sha256((proof_path.with_suffix('') / r['file']).read_bytes()).hexdigest() == r['sha256']
receipt('optimize-log-bf16-v10')
exhaustive = json.loads((HERE / 'edges/log-bf16-v10-exhaustive.json').read_text())
assert len(exhaustive) == 1 and exhaustive[0]['passed'] and exhaustive[0]['all_bit_patterns']
assert exhaustive[0]['source_sha256'] == candidate_hash

selected = json.loads((HERE / 'proposal-v5-configs.json').read_text())
qualified = json.loads((HERE / 'qualified-baselines.json').read_text())
for row in selected:
    if row['op'] == 'LogFwdOp' and row['case_id'].endswith('bfloat16'):
        tag = 'public-log-bf16-v10-' + row['case_id']
        receipt(tag)
        trials = json.loads((HERE / 'search' / tag / 'trials.json').read_text())
        assert len(trials) == 3 and all(t['status'] == 'ok' and t['correct'] and t['input_unchanged'] for t in trials)
        for trial in trials:
            assert trial['warmup'] == 3 and trial['repeats'] == 15 and not trial['event_used']
            samples = trial['profile']['latency_samples_ms']
            assert len(samples) == 15 and abs(sum(samples) / 15 - trial['profile']['latency_ms']) < 1e-12
        best = min(trials, key=lambda t: t['profile']['latency_ms'])
        opponents = [b for b in qualified if b['op'] == row['op'] and b['case_id'] == row['case_id']]
        assert opponents and all(b['input_digest'] == best['input_digest'] for b in opponents)
        assert min(b['latency_ms'] for b in opponents) / best['profile']['latency_ms'] >= .8
        row.update(config=best['config'], comparison_us=best['profile']['latency_ms'] * 1000,
                   evidence='search/' + tag, guard_request=tag,
                   source_eligibility={'mode': 'identical_candidate_helper_hash', 'helper_sha256': candidate_hash},
                   qualification='public candidate configuration selection only; frozen-source final M4 required')
    else:
        matches = [r for r in proof['comparisons'] if r['op'] == row['op'] and r['case_id'] == row['case_id'] and r['config'] == row['config']]
        assert len(matches) == 1
        row['source_eligibility'] = {
            'mode': 'exact_selected_generated_C_after_Log_BF16_change',
            'proof': str(proof_path.relative_to(HERE)),
            'prior_eligibility': row['source_eligibility'],
            'old_helper_sha256': proof['old_helper_sha256'], 'new_helper_sha256': candidate_hash,
        }

snapshot = HERE / 'source-v5'
snapshot.mkdir(exist_ok=False)
for op, kind in [('LogFwdOp', 'log'), ('Log1pFwdOp', 'log1p'), ('SqrtFwdOp', 'sqrt'), ('RsqrtFwdOp', 'rsqrt')]:
    path = KERNELS / (kind + '.py')
    source = path.read_text()
    for row in selected:
        if row['op'] != op:
            continue
        size, dtype = row['case_id'].split('-')
        n = int(size.removesuffix('M')) * 1024 * 1024
        prefix = f'        ({n}, {dtype!r}): '
        start = source.index(prefix)
        end = source.index('\n', start)
        config = tuple(row['config'][k] for k in ('tile', 'blocks', 'stages'))
        source = source[:start] + prefix + repr(config) + ',' + source[end:]
    path.write_text(source)
shutil.copy2(candidate, KERNELS / '_logroots.py')
files = [KERNELS / (name + '.py') for name in ('_logroots', 'log', 'log1p', 'sqrt', 'rsqrt')]
frozen = dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              sources={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
              reason='Actual public-entry FP32 configurations; BF16 Log normalizes before a single hardware ln, with exhaustive value validation and 18 unchanged selected C kernels.')
for path in files:
    shutil.copy2(path, snapshot / path.name)
(HERE / 'frozen-v5.json').write_text(json.dumps(frozen, indent=2) + '\n')
(HERE / 'selected-configs-v5.json').write_text(json.dumps(selected, indent=2) + '\n')
(HERE / 'CURRENT-FREEZE.json').write_text(json.dumps({'file': 'frozen-v5.json', 'selected': 'selected-configs-v5.json'}, indent=2) + '\n')
print(json.dumps(frozen, indent=2))
