"""CPU-only accounting from the existing exact-B1 first-worker authorization."""
from pathlib import Path
import json
import shutil


def effective_available(host_available, limit, usage, stats, v2=False):
    names = ('inactive_file', 'active_file', 'shmem', 'file_dirty', 'file_writeback', 'unevictable') if v2 else ('total_inactive_file', 'total_active_file', 'total_shmem', 'total_dirty', 'total_writeback', 'total_unevictable')
    inactive, active, shmem, dirty, writeback, unevictable = (stats[n] for n in names)
    clean_inactive = max(0, inactive - dirty - writeback)
    clean_active = max(0, active - shmem - dirty - writeback - unevictable)
    raw = max(0, limit - usage)
    # Apply the authorized formula before clamping the final estimate.
    estimate = max(0, min(host_available, limit - usage + clean_inactive + clean_active // 2))
    return {'effective_available_host_bytes': estimate, 'raw_headroom_bytes': raw,
            'clean_inactive_bytes': clean_inactive, 'clean_active_bytes': clean_active}


def capacity():
    host = next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
    roots = [(Path('/sys/fs/cgroup/memory'), 'memory.limit_in_bytes', 'memory.usage_in_bytes', False),
             (Path('/sys/fs/cgroup'), 'memory.max', 'memory.current', True)]
    for root, limit_name, usage_name, v2 in roots:
        if (root/limit_name).is_file() and (root/usage_name).is_file():
            limit = (root/limit_name).read_text().strip()
            usage = int((root/usage_name).read_text())
            stats = {k: int(v) for k, v in (line.split() for line in (root/'memory.stat').read_text().splitlines())}
            observed = ({'effective_available_host_bytes': host} if limit == 'max' else effective_available(host, int(limit), usage, stats, v2))
            return dict(observed, cgroup_limit=limit, cgroup_usage=usage, cgroup_mount=str(root), memory_stat=stats, host_available_bytes=host)
    raise RuntimeError('Cannot establish current effective host/cgroup capacity')


def require_capacity(here, run_dir):
    estimates = json.loads((here/'RESOURCE_ESTIMATES.json').read_text())
    case_estimates = json.loads((here.parent/'CASE_RESOURCE_ESTIMATES.json').read_text())[run_dir.name]
    estimates = dict(estimates, **case_estimates)
    observed = capacity()
    observed['required_envelope'] = case_estimates
    observed['disk_free_bytes'] = shutil.disk_usage(run_dir).free
    (run_dir/'fresh_resource_admission.json').write_text(json.dumps(observed, indent=2))
    if observed['effective_available_host_bytes'] < estimates['minimum_effective_available_host_bytes'] or observed['disk_free_bytes'] < estimates['minimum_free_disk_bytes']:
        raise RuntimeError('Full case resource envelope unavailable: ' + json.dumps(observed))
    return observed
