"""Freeze a BF16 lookup only after exhaustive and valid public timing gates."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
KERNELS = ROOT / 'TileOPs/src/tileops/kernels/ascend/elementwise'
p = argparse.ArgumentParser()
p.add_argument('--candidate', required=True)
p.add_argument('--version', required=True)
a = p.parse_args()


def read(path):
    return json.loads(path.read_text())


def receipt(label):
    values = [read(path) for path in (HERE / 'requests').glob(label + '_*/status.json')]
    value = max(values, key=lambda row: row['completed_at'])
    assert value['guard_exit_code'] == 0, value


current = read(HERE / 'CURRENT-FREEZE.json')
frozen = read(HERE / current['file'])
assert all(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
           for path, digest in frozen['sources'].items())
candidate = HERE / ('candidate/log_bf16_lut_' + a.candidate + '.py')
candidate_hash = hashlib.sha256(candidate.read_bytes()).hexdigest()
receipt('optimize-log-bf16-lut-' + a.candidate)
probe = read(HERE / ('edges/log-lut-' + a.candidate + '-exhaustive.json'))
assert probe['source_sha256'] == candidate_hash and probe['exact_CPU_values']
assert probe['all_65536_patterns'] and probe['extra_tail']
selected = read(HERE / current['selected'])
opponents = read(HERE / 'qualified-baselines.json')
configs = {}
for case in ('16M-bfloat16', '256M-bfloat16'):
    label = 'public-log-bf16-lut-' + a.candidate + '-' + case
    receipt(label)
    directory = HERE / 'search' / label
    sources = read(directory / 'sources.json')
    assert candidate_hash in sources.values()
    trials = read(directory / 'trials.json')
    assert len(trials) == 3 and all(t['status'] == 'ok' and t['correct'] and t['input_unchanged'] for t in trials)
    for t in trials:
        assert t['warmup'] == 3 and t['repeats'] == 15 and not t['event_used']
        samples = t['profile']['latency_samples_ms']
        assert len(samples) == 15 and abs(sum(samples) / 15 - t['profile']['latency_ms']) < 1e-12
    best = min(trials, key=lambda t: t['profile']['latency_ms'])
    baselines = [b for b in opponents if b['op'] == 'LogFwdOp' and b['case_id'] == case]
    assert baselines and all(b['input_digest'] == best['input_digest'] for b in baselines)
    assert min(b['latency_ms'] for b in baselines) / best['profile']['latency_ms'] >= .8
    configs[case] = tuple(best['config'][k] for k in ('tile', 'blocks', 'stages'))
    row = next(r for r in selected if r['op'] == 'LogFwdOp' and r['case_id'] == case)
    row.update(config=best['config'], comparison_us=best['profile']['latency_ms'] * 1000,
               evidence='search/' + label, guard_request=label,
               source_eligibility={'mode': 'exhaustive_BF16_lookup_with_public_configuration_search',
                                   'candidate_sha256': candidate_hash, 'table_sha256': probe['table_sha256']},
               qualification='Selected candidate only; integrated production entry requires final M3/M4')

path = KERNELS / 'log.py'
template = HERE / 'source-v6/log.py'
template_digest = read(HERE / 'frozen-v6.json')['sources'][str(path.relative_to(ROOT))]
assert hashlib.sha256(template.read_bytes()).hexdigest() == template_digest
source = template.read_text()
registration = '\n\nif T is not None:\n    from tileops.backend'
assert source.count(registration) == 1
for case, config in configs.items():
    n = int(case.split('M-')[0]) * 1024 * 1024
    pattern = r'\(' + str(n) + r', [\x22\x27]bfloat16[\x22\x27]\): \([^\n]*\),'
    source, count = re.subn(pattern, f'({n}, "bfloat16"): {config},', source)
    assert count == 1
source, count = re.subn(r'(?m)^(        [\x22\x27]bfloat16[\x22\x27]: )\([^\n]*\),',
                        lambda m: m[1] + repr(configs['16M-bfloat16']) + ',', source)
assert count == 1
lookup = candidate.read_text().split('def table(', 1)[1]
lookup = '@functools.lru_cache(maxsize=8)\ndef table(' + lookup
for old, new in [('table', '_lookup_table'), ('body', '_lookup_body'), ('build', '_build_lookup')]:
    lookup = re.sub(r'\b' + old + r'\b', new, lookup)
lookup = lookup.replace('dtype=torch.int32)', "dtype=torch.int32, device='cpu')")
lookup = lookup.replace('    assert n > 0 and tile > 0 and tile % 128 == 0',
                        "    if n <= 0 or tile <= 0 or tile % 128 or blocks <= 0 or stages <= 0:\n        raise ValueError('Expected positive size/grid and a tile divisible by 128')")
lookup = lookup.replace('        assert max_stages >= 1',
                        "        if max_stages < 1:\n            raise ValueError('Lookup, tile and tail buffers exceed available UB capacity')")
lookup = lookup.replace('            if tail:', '            if tail:  # noqa: SIM102 - static guard excludes zero-size TIR buffers')
source = source.replace('from tileops.kernels.ascend', 'import functools\n\nimport torch\n\nfrom tileops.kernels.ascend', 1)
source = source.replace('class AscendLogKernel', 'if T is not None:\n    import tilelang\n\n' +
                        ''.join('    ' + line if line.strip() else line for line in lookup.splitlines(keepends=True)) +
                        '\n\nclass AscendLogKernel', 1)
methods = '''
    def __init__(self, n_total, dtype):
        if dtype != torch.bfloat16:
            super().__init__(n_total, dtype)
            return
        super().__init__(0, dtype)
        self.n_total = n_total
        if n_total:
            config = self.configs.get((n_total, "bfloat16"), self.configs["bfloat16"])
            self.compiled = _build_lookup(n_total, *config)
            self.lookup = _lookup_table(torch.device("npu", torch.npu.current_device()))

    def forward(self, input):
        if self.dtype != torch.bfloat16 or not self.n_total:
            return super().forward(input)
        if input.device.type != "npu" or input.dtype != self.dtype or input.numel() != self.n_total:
            raise ValueError("Input must match the NPU kernel dtype and element count")
        return self.compiled(input.reshape(-1), self.lookup).reshape(input.shape)
'''
source = source.replace(registration, '\n' + methods + registration)
staged = HERE / ('candidate/log_module_' + a.version + '.py')
assert not staged.exists() and not (HERE / ('source-' + a.version)).exists()
staged.write_text(source)
ruff = '/root/dyq/workspace/.venv/bin/ruff'
config_path = str(ROOT / 'TileOPs/pyproject.toml')
subprocess.run([ruff, 'check', '--config', config_path, '--fix', str(staged)], check=True)
subprocess.run([ruff, 'format', '--config', config_path, str(staged)], check=True)
compile(staged.read_text(), str(staged), 'exec')
# All gates and static checks precede production writes.
assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
           for name, digest in frozen['sources'].items())
snapshot = HERE / ('source-' + a.version)
snapshot.mkdir(exist_ok=False)
shutil.copy2(staged, path)
frozen['sources'] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in frozen['sources']}
frozen.update(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              reason='BF16 Log uses a complete input-independent value table and NPU SIMD gather. Other three operators and shared helper are unchanged.',
              lookup_candidate=a.candidate, lookup_table_sha256=probe['table_sha256'])
for name in frozen['sources']:
    shutil.copy2(ROOT / name, snapshot / Path(name).name)
(HERE / ('frozen-' + a.version + '.json')).write_text(json.dumps(frozen, indent=2) + '\n')
(HERE / ('selected-configs-' + a.version + '.json')).write_text(json.dumps(selected, indent=2) + '\n')
(HERE / 'CURRENT-FREEZE.json').write_text(json.dumps(dict(file='frozen-' + a.version + '.json', selected='selected-configs-' + a.version + '.json'), indent=2) + '\n')
print(json.dumps(frozen, indent=2))
