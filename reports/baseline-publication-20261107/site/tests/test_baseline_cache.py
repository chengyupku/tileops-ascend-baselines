import json

import pytest
import render_baseline_cache as cache


def test_baseline_only_rows_keep_missing_tileops_and_hide_host(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, 'ROOT', tmp_path)
    source = tmp_path / 'data/baseline-cache'
    source.mkdir(parents=True)
    row = dict(library='ops-nn', op='ReluFwdOp', case='small-f16',
               latency_us=12.5, samples=15,
               ratio=None, tileops_latency_us=None, workload={'family': 'elementwise'})
    snapshot = dict(frozen_at='2026-10-29', manifest_cases=2062, rows=[row])
    (source / 'snapshot.json').write_text(json.dumps(snapshot))
    cache.render()
    out = tmp_path / 'docs/benchmarks'
    text = (out / 'elementwise.md').read_text()
    assert '<th colspan="2">Alternatives</th>' in text
    assert '<th>Device time</th>' in text
    assert '<th>Throughput</th>' in text
    assert 'server2' not in text
    assert '<td>NaN</td><td>—</td><td><code>ops-nn</code></td>' in text
    assert '<td>0.0125</td>' in text
    assert '12.500 D' not in text
    assert '1 / 2062 workloads' in (out / 'index.md').read_text()
    before = (out / 'index.md').read_bytes()
    snapshot['rows'] = [row, row]
    (source / 'snapshot.json').write_text(json.dumps(snapshot))
    with pytest.raises(AssertionError, match='Duplicate'):
        cache.render()
    assert (out / 'index.md').read_bytes() == before


def test_alternatives_stack_and_private_provenance_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, 'ROOT', tmp_path)
    source = tmp_path / 'data/baseline-cache'
    source.mkdir(parents=True)
    base = dict(op='AddFwdOp', case='small-float16',
                samples=15, ratio=None,
                tileops_latency_us=None, workload={'family': 'elementwise'})
    rows = [{**base, 'library': 'fast', 'latency_us': 10},
            {**base, 'library': 'slow', 'latency_us': 20}]
    (source / 'snapshot.json').write_text(json.dumps(
        dict(frozen_at='2026-10-29', manifest_cases=2062, rows=rows)))
    cache.render()
    text = (tmp_path / 'docs/benchmarks/elementwise.md').read_text()
    assert text.count('<table>') == 1
    assert 'Host round-trip time' not in text
    assert '<td class="colsep">f16</td>' in text
    assert '<code>fast</code><br><span class="alt-slow"><code>slow</code>' in text
    assert '<td>0.0100<br><span class="alt-slow">0.0200</span></td>' in text
    assert text.count('<td>NaN</td>') == 1
    snapshot = dict(frozen_at='2026-10-29', manifest_cases=2062,
                    rows=[{**base, 'library': 'bad', 'latency_us': 100,
                           'metric': 'end_to_end_mean_ms'}])
    (source / 'snapshot.json').write_text(json.dumps(snapshot))
    with pytest.raises(AssertionError, match='provenance'):
        cache.render()
    assert (tmp_path / 'docs/benchmarks/elementwise.md').read_text() == text
    snapshot['rows'][0].pop('metric')
    snapshot['rows'][0].update(host='server1')
    (source / 'snapshot.json').write_text(json.dumps(snapshot))
    with pytest.raises(AssertionError, match='provenance'):
        cache.render()


@pytest.mark.parametrize(('family', 'slug'), [
    ('gemm', 'gemm'), ('norm', 'normalization'),
    ('rope', 'rope'), ('mamba', 'ssm'),
])
def test_manifest_family_aliases_reach_their_pages(tmp_path, monkeypatch, family, slug):
    monkeypatch.setattr(cache, 'ROOT', tmp_path)
    source = tmp_path / 'data/baseline-cache'
    source.mkdir(parents=True)
    row = dict(library='baseline', op='ExampleOp', case='small-float16',
               latency_us=10, samples=15, ratio=None, tileops_latency_us=None,
               workload={'family': family})
    (source / 'snapshot.json').write_text(json.dumps(
        dict(frozen_at='2026-10-29', manifest_cases=1, rows=[row])))
    cache.render()
    out = tmp_path / 'docs/benchmarks'
    assert 'baseline' in (out / (slug + '.md')).read_text()
    assert f']({slug}.md)' in (out / 'index.md').read_text()
    assert not (out / 'other.md').exists()


def test_verified_tileops_uses_fastest_baseline_and_rejects_inconsistent_ratio(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, 'ROOT', tmp_path)
    source = tmp_path / 'data/baseline-cache'
    source.mkdir(parents=True)
    base = dict(op='AddFwdOp', case='small-float16', samples=15,
                tileops_latency_us=5, tileops_samples=15, tileops_verified=True,
                workload={'family': 'elementwise'})
    rows = [{**base, 'library': 'slow', 'latency_us': 20, 'ratio': 4},
            {**base, 'library': 'fast', 'latency_us': 10, 'ratio': 2}]
    snapshot = dict(frozen_at='2026-11-01', manifest_cases=1, rows=rows)
    (source / 'snapshot.json').write_text(json.dumps(snapshot))
    cache.render()
    text = (tmp_path / 'docs/benchmarks/elementwise.md').read_text()
    assert '<td><span class="perf-ahead">2.00×</span></td><td>0.005</td>' in text
    assert text.index('<code>fast</code>') < text.index('<code>slow</code>')
    rows[0]['ratio'] = 40
    (source / 'snapshot.json').write_text(json.dumps(snapshot))
    with pytest.raises(AssertionError):
        cache.render()
    assert (tmp_path / 'docs/benchmarks/elementwise.md').read_text() == text


def test_verified_tileops_without_eligible_baseline_keeps_nan(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, 'ROOT', tmp_path)
    source = tmp_path / 'data/baseline-cache'
    source.mkdir(parents=True)
    snapshot = dict(frozen_at='2026-11-01', manifest_cases=1, rows=[], tileops_rows=[
        dict(op='SinusoidalFwdOp', case='small-float32', latency_us=5,
             samples=15, verified=True, workload={'family': 'elementwise'})])
    (source / 'snapshot.json').write_text(json.dumps(snapshot))
    cache.render()
    text = (tmp_path / 'docs/benchmarks/elementwise.md').read_text()
    assert '<td>NaN</td><td>0.005</td><td>—</td><td>—</td>' in text
    index = (tmp_path / 'docs/benchmarks/index.md').read_text()
    assert '0 baseline measurements' in index
    assert 'Verified TileOPs results: 1 operators · 1 workloads' in index


def test_different_workload_contracts_are_rejected_before_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, 'ROOT', tmp_path)
    source = tmp_path / 'data/baseline-cache'
    source.mkdir(parents=True)
    row = dict(library='first', op='ExampleOp', case='same-id-float16',
               latency_us=10, samples=15, ratio=None, tileops_latency_us=None,
               workload={'family': 'elementwise', 'tensors': {'x': [[32], 'float16']}})
    snapshot = dict(frozen_at='2026-11-01', manifest_cases=2208, rows=[row])
    (source / 'snapshot.json').write_text(json.dumps(snapshot))
    cache.render()
    output = tmp_path / 'docs/benchmarks/elementwise.md'
    original = output.read_bytes()
    snapshot['rows'].append({**row, 'library': 'second',
                             'workload': {'family': 'elementwise', 'tensors': {'x': [[64], 'float16']}}})
    (source / 'snapshot.json').write_text(json.dumps(snapshot))
    with pytest.raises(AssertionError, match='workload contract'):
        cache.render()
    assert output.read_bytes() == original


@pytest.mark.parametrize(('library', 'baseline_us', 'css'), [
    ('torch-eager', 10, 'perf-unrated'),
    ('native-library', 3, 'perf-behind'),
])
def test_ratio_uses_existing_verdict_style(tmp_path, monkeypatch, library, baseline_us, css):
    monkeypatch.setattr(cache, 'ROOT', tmp_path)
    source = tmp_path / 'data/baseline-cache'
    source.mkdir(parents=True)
    row = dict(library=library, op='ExampleOp', case='small-float16',
               latency_us=baseline_us, samples=15, ratio=baseline_us/5,
               tileops_latency_us=5, tileops_samples=15, tileops_verified=True,
               workload={'family': 'elementwise'})
    (source / 'snapshot.json').write_text(json.dumps(
        dict(frozen_at='2026-11-01', manifest_cases=1, rows=[row])))
    cache.render()
    text = (tmp_path / 'docs/benchmarks/elementwise.md').read_text()
    assert f'<span class="{css}">{baseline_us/5:.2f}×</span>' in text


def test_published_torch_survives_older_nightly_and_keeps_tileops():
    workload = {'params': {'n': 32}, 'family': 'elementwise'}
    prior = dict(library='native', op='AddFwdOp', case='small', workload=workload,
                 latency_us=20, tileops_latency_us=5, tileops_samples=15,
                 tileops_verified=True, tileops_measurement_id='tile', tileops_protocol='v3')
    new = dict(library='torch-eager', op='AddFwdOp', case='small', workload=workload,
               latency_us=10, tileops_latency_us=None, ratio=None,
               measurement_id='new', protocol='v3')
    official = dict(rows=[prior], frozen_at='2026-11-02')
    published = dict(rows=[new], frozen_at='2026-11-04', publication_notes=['new release'])
    release = dict(count=1, measurements=[dict(op='AddFwdOp', case='small',
                                             measurement_id='new', protocol='v3')])
    merged = cache.merge_torch_release(official, published, release)
    assert merged['rows'][0] == prior
    row = merged['rows'][1]
    assert row['latency_us'] == 10 and row['tileops_latency_us'] == 5
    assert row['ratio'] == 2 and row['tileops_measurement_id'] == 'tile'
    assert merged['publication_notes'] == ['new release']
    prior['workload'] = {'params': {'n': 64}, 'family': 'elementwise'}
    with pytest.raises(AssertionError, match='workload mismatch'):
        cache.merge_torch_release(official, published, release)


def test_new_release_promotes_tile_only_infinity_without_measurement_id():
    work = {'params': {'ord': 'Infinity', 'dim': -1}, 'family': 'reduction'}
    tile = dict(op='VectorNormFwdOp', case='inf', workload={
        'params': {'ord': float('inf'), 'dim': -1}, 'family': 'reduction'},
        latency_us=5, samples=15, verified=True, protocol='v3', commit='source')
    new = dict(library='torch-eager', op=tile['op'], case=tile['case'], workload=work,
               latency_us=10, tileops_latency_us=None, ratio=None,
               measurement_id='new', protocol='v3')
    official = dict(rows=[], tileops_rows=[tile], frozen_at='2026-11-05')
    published = dict(rows=[new], frozen_at='2026-11-06')
    release = dict(count=1, measurements=[dict(op=tile['op'], case=tile['case'],
                                             measurement_id='new', protocol='v3')])
    result = cache.merge_torch_release(official, published, release)
    assert result['tileops_rows'] == []
    assert result['rows'][0]['ratio'] == 2
    assert result['rows'][0]['tileops_commit'] == 'source'
    assert result['rows'][0]['workload'] == work
    tile['workload']['params']['dim'] = 0
    with pytest.raises(AssertionError, match='workload mismatch'):
        cache.merge_torch_release(official, published, release)
