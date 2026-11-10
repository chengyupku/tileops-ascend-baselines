from pathlib import Path
import subprocess,shlex
P=Path(__file__).resolve().parent;R='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_bmm1';SSH='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
s=(P/'resource_gate.py').read_text();s=s.replace('def capacity(proc=Path', 'def capacity_v2(proc=Path')
s+='''

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
'''
(P/'resource_gate.py').write_text(s)
archive=f'''from pathlib import Path
import shutil
p=Path({R!r});a=p/'attempt1_capacity_v2';a.mkdir()
for name in ['guard.log','guard.exit','GUARD_FINAL.json','FROZEN.json','resource_gate.py']:
 shutil.copy2(p/name,a/name)
shutil.copytree(p/'square-4k-bfloat16',a/'case')
'''
subprocess.run([SSH,'openlibing-DevEnv-661495','python3 -c '+shlex.quote(archive)],check=True)
subprocess.run([SSH,'openlibing-DevEnv-661495','cat > '+R+'/resource_gate.py'],input=s.encode(),check=True)
script=f'''from pathlib import Path
import json,hashlib,runpy
p=Path({R!r});data=json.loads((p/'FROZEN.json').read_text());f=p/'resource_gate.py';data['sha256'][str(f)]=hashlib.sha256(f.read_bytes()).hexdigest();(p/'FROZEN.json').write_text(json.dumps(data,indent=2));d=runpy.run_path(str(f))['capacity']();(p/'CPU_CAPACITY.json').write_text(json.dumps(d,indent=2));print(d['effective_available_host_bytes']);assert d['effective_available_host_bytes']>=10200547328
'''
subprocess.run([SSH,'openlibing-DevEnv-661495','python3 -c '+shlex.quote(script)],check=True)
