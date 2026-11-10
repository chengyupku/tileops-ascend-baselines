"""Retrieve this task's output directories over the available SSH transport."""
from pathlib import Path
import subprocess
import sys
import shlex
import tarfile
HERE=Path(__file__).resolve().parent
allowed={'initial-log.jsonl','probe-v1.json','probe-v2.json','baseline','environment','search','final','profiles','edges','probe-v3-f16.json','probe-v3-bf16.json'}
assert set(sys.argv[1:])<=allowed
command=['tar','-c','-C','/opt/tileops-all-pairs/workspace/docs/ascend950/dev/logroots4_20261028',*sys.argv[1:]]
p=subprocess.Popen(['ssh','-o','ClearAllForwardings=yes','openlibing-DevEnv-661495',shlex.join(command)],stdout=subprocess.PIPE)
with tarfile.open(fileobj=p.stdout,mode='r|') as archive:
    for member in archive:
        if not member.isdir() and not member.isfile():raise RuntimeError('Unexpected archive member')
        if not (HERE/member.name).resolve().is_relative_to(HERE):raise RuntimeError('Unsafe member')
        archive.extract(member,HERE,filter='data')
assert p.wait()==0
