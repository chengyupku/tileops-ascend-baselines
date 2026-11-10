"""Keep published baselines authoritative while joining only matching TileOPs contracts."""
import copy
import json
import math


def contract(row):
    def normalize(value):
        if isinstance(value, float) and not math.isfinite(value):
            return 'NaN' if math.isnan(value) else ('Infinity' if value > 0 else '-Infinity')
        if isinstance(value, dict):
            return {k: normalize(v) for k, v in value.items()}
        if isinstance(value, list):
            return [normalize(v) for v in value]
        return value
    return json.dumps(normalize(row['workload']['params']), sort_keys=True)


def identity(row):
    return row['op'], row['case'], contract(row)


def merge_latest_release(official, published, release):
    """An older nightly may add TileOPs evidence, never roll back a baseline release."""
    data = copy.deepcopy(published)
    actual = {(r['library'], r['op'], r['case']): r for r in data['rows']}
    assert len(actual) == release['count'] == len(release['measurements'])
    for item in release['measurements']:
        row = actual[(item['library'], item['op'], item['case'])]
        assert all(row[k] == item[k] for k in ('measurement_id', 'protocol'))
    tiles = {}
    for row in official['rows'] + official.get('historical_rows', []):
        if row.get('tileops_latency_us') is not None:
            tiles[identity(row)] = {k: v for k, v in row.items() if k.startswith('tileops_')}
    for tile in official.get('tileops_rows', []) + official.get('historical_tileops_rows', []):
        tiles[identity(tile)] = {'tileops_' + k: v for k, v in tile.items()
                                 if k in ('latency_us', 'samples', 'verified', 'protocol',
                                          'measurement_id', 'commit', 'date')}
    for row in data['rows'] + data.get('historical_rows', []):
        candidate = tiles.get(identity(row))
        if candidate and (not row.get('tileops_date') or
                          candidate.get('tileops_date', '') >= row['tileops_date']):
            row.update(candidate)
            row['ratio'] = row['latency_us'] / row['tileops_latency_us']
    visible = {(r['op'], r['case']) for r in data['rows']}
    extra = {(r['op'], r['case']): r for r in data.get('tileops_rows', [])}
    for row in official.get('tileops_rows', []):
        if (row['op'], row['case']) not in visible:
            extra[(row['op'], row['case'])] = row
    data['tileops_rows'] = list(extra.values())
    data['historical_tileops_rows'] = official.get('historical_tileops_rows', [])
    if 'nightly' in official:
        data['nightly'] = official['nightly']
    data['frozen_at'] = max(official['frozen_at'], published['frozen_at'])
    return data
