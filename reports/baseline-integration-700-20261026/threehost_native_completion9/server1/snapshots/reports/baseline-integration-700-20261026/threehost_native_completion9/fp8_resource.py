from pathlib import Path

def capacity():
 p=Path('/sys/fs/cgroup/memory')
 stats=dict((k,int(v)) for k,v in (line.split() for line in (p/'memory.stat').read_text().splitlines()))
 limit=min(int((p/'memory.limit_in_bytes').read_text()),stats['hierarchical_memory_limit'])
 usage=int((p/'memory.usage_in_bytes').read_text())
 host=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
 return {'effective_available_host_bytes':min(host,max(0,limit-usage)),'host_available_bytes':host,'limit':limit,'usage':usage,'basis':'visible cgroup-v1 raw headroom, hierarchical limit; no reclaim credit','ancestor_usage_visibility':'container mount; ancestor usage unavailable'}
