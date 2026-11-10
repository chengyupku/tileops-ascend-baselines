"""Read-only cgroup-v2 accounting proposal; never imports a device library."""
import json
import pathlib
import shutil
import sys


def probe(disk_path):
    host = next(int(l.split()[1])*1024 for l in pathlib.Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))
    membership = [l.split(':', 2)[2] for l in pathlib.Path('/proc/self/cgroup').read_text().splitlines() if l.startswith('0::')]
    mounts = [l.split() for l in pathlib.Path('/proc/self/mountinfo').read_text().splitlines() if ' - cgroup2 ' in l]
    if len(membership) != 1 or len(mounts) != 1 or mounts[0][3] != '/':
        raise RuntimeError('Cannot attest full visible cgroup-v2 ancestor chain')
    root = pathlib.Path(mounts[0][4]); current = root / membership[0].lstrip('/')
    values = [host]; ancestors = []
    while True:
        row = {'path': str(current)}
        if current == root:
            if (current/'memory.max').exists():
                raise RuntimeError('Unexpected root memory controller layout; review required')
            row['unlimited_root'] = True
        else:
            limit = (current/'memory.max').read_text().strip()
            usage = int((current/'memory.current').read_text())
            stats = dict((k, int(v)) for k,v in (l.split() for l in (current/'memory.stat').read_text().splitlines()))
            row.update(limit=limit, usage=usage, memory_high=(current/'memory.high').read_text().strip(), stats=stats)
            if limit != 'max':
                inactive=max(0,stats['inactive_file']-stats['file_dirty']-stats['file_writeback'])
                active=max(0,stats['active_file']-stats['shmem']-stats['file_dirty']-stats['file_writeback']-stats['unevictable'])
                value=max(0,min(host,int(limit)-usage+inactive+active//2))
                row['effective_available_host_bytes']=value; values.append(value)
        ancestors.append(row)
        if current == root: break
        current=current.parent
    return {'host_available_bytes':host,'effective_available_host_bytes':min(values),'ancestors':ancestors,'disk_free_bytes':shutil.disk_usage(disk_path).free,'limitation':'Visible root mount required; deployment supervisor must rule out hidden ancestor limits if entering a different namespace. Reprobe inside guard child.'}

if __name__ == '__main__':
    print(json.dumps(probe(sys.argv[1] if len(sys.argv)>1 else '.'),indent=2))
