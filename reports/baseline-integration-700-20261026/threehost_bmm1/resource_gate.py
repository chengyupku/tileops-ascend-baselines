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


def capacity_v2(proc=Path('/proc')):
    host = next(int(l.split()[1])*1024 for l in (proc/'meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))
    membership = [l.split(':',2)[2] for l in (proc/'self/cgroup').read_text().splitlines() if l.startswith('0::')]
    mounts = [l.split() for l in (proc/'self/mountinfo').read_text().splitlines() if ' - cgroup2 ' in l]
    if len(membership)!=1 or len(mounts)!=1 or mounts[0][3]!='/':
        raise RuntimeError('Cannot attest full visible cgroup-v2 ancestors')
    def normalized(s):
        p=Path(s)
        if not p.is_absolute() or '..' in p.parts or str(p)!=s or '\\' in s:
            raise RuntimeError('Noncanonical cgroup path')
        return p
    member=normalized(membership[0]);root=normalized(mounts[0][4])
    current=root/member.relative_to('/')
    if current.resolve()!=current or root.resolve()!=root:
        raise RuntimeError('Invisible or symlinked cgroup ancestor')
    values=[host]; rows=[]
    while True:
        if not current.is_dir():raise RuntimeError('Invisible cgroup ancestor')
        row={'path':str(current)}
        if current==root:
            if (current/'memory.max').exists():raise RuntimeError('Unexpected root memory controller layout')
            row['unlimited_root']=True
        else:
            limit=(current/'memory.max').read_text().strip()
            high=(current/'memory.high').read_text().strip()
            usage=int((current/'memory.current').read_text())
            stats=dict((k,int(v)) for k,v in (l.split() for l in (current/'memory.stat').read_text().splitlines()))
            if usage<0 or any(v<0 for v in stats.values()):raise RuntimeError('Invalid memory accounting')
            for name in ('inactive_file','active_file','shmem','file_dirty','file_writeback','unevictable'):stats[name]
            finite=[]
            for v in (limit,high):
                if v!='max':
                    if not v.isdecimal():raise RuntimeError('Invalid cgroup limit')
                    finite.append(int(v))
            row.update(limit=limit,memory_high=high,usage=usage,stats=stats)
            if finite:
                row.update(effective_available(host,min(finite),usage,stats,True))
                values.append(row['effective_available_host_bytes'])
        rows.append(row)
        if current==root:break
        current=current.parent
    return {'host_available_bytes':host,'effective_available_host_bytes':min(values),'ancestors':rows}


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


def capacity(proc=Path('/proc')):
    if any(l.startswith('0::') for l in (proc/'self/cgroup').read_text().splitlines()):
        return capacity_v2(proc)
    member=next(l.split(':',2)[2] for l in (proc/'self/cgroup').read_text().splitlines() if 'memory' in l.split(':')[1].split(','))
    fields=next(l.split() for l in (proc/'self/mountinfo').read_text().splitlines() if ' - cgroup ' in l and 'memory' in l.split()[-1].split(','))
    assert fields[3]==member, 'Memory mount does not match current cgroup'
    root=Path(fields[4]);stats=dict((k,int(v)) for k,v in (line.split() for line in (root/'memory.stat').read_text().splitlines()))
    host=next(int(l.split()[1])*1024 for l in (proc/'meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))
    limit=min(int((root/'memory.limit_in_bytes').read_text()),stats['hierarchical_memory_limit'])
    high=(root/'memory.high').read_text().strip()
    if high!='max':limit=min(limit,int(high))
    usage=int((root/'memory.usage_in_bytes').read_text())
    # Conservative raw headroom; do not rely on reclaimable caches.
    return dict(host_available_bytes=host,effective_available_host_bytes=min(host,max(0,limit-usage)),cgroup_version=1,member=member,mount=fields,limit=limit,usage=usage,stats=stats,accounting='raw headroom capped by reported hierarchical limit; no reclaim credit')
