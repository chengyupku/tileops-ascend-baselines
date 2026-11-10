from control import *
import tarfile,hashlib
r=py(f'''from pathlib import Path
import json,hashlib,shutil,re
p=Path({R!r});d=p/'device_proof';d.mkdir()
lines=(p/'DEVICE_LOG_EXTRACT.txt').read_text().splitlines()
selected=[l for l in lines if 'GroupedMatmul' in l and any(t in l for t in ['Available bin','Register Kernel succ','StreamLaunchKernelV2'])]
assert len(selected)==3,selected
binpath=Path(re.search(r'bin path: (\\S+)',selected[1]).group(1))
assert binpath.is_relative_to(p/'private_provider')
frozen=json.loads((p/'FROZEN.json').read_text())['sha256']
records=[]
for src in [binpath,binpath.with_suffix('.json'),p/'private_provider/op_api/lib/libcust_opapi.so']:
 raw=src.read_bytes();h=hashlib.sha256(raw).hexdigest();assert h==frozen[str(src)]
 shutil.copy2(src,d/src.name)
 records.append(dict(path=str(src),sha256=h,bytes=len(raw),inode=src.stat().st_ino,elf_machine=int.from_bytes(raw[18:20],'little') if raw[:4]==b'\\x7fELF' else None))
(d/'PROOF.json').write_text(json.dumps(dict(launch_log_lines=selected,files=records),indent=2))
shutil.copytree(p/'square-4k-bfloat16/ASCEND_PROCESS_LOG_PATH',d/'process_logs')
print(json.dumps(dict(files=records)))
''');assert r['rc']==0,r
(P/'DEVICE_PROOF_RECEIPT.json').write_text(json.dumps(r,indent=2))
with (P/'device_proof.tar.gz').open('wb') as out:
 t=subprocess.run([SSH,HOST,f'tar -C {R} -czf - device_proof'],stdout=out,stderr=subprocess.PIPE);assert t.returncode==0,t.stderr
with tarfile.open(P/'device_proof.tar.gz') as t:t.extractall(P,filter='data')
x=json.loads((P/'device_proof/PROOF.json').read_text())
for row in x['files']:assert hashlib.sha256((P/'device_proof'/Path(row['path']).name).read_bytes()).hexdigest()==row['sha256']
print('device binary, host ELF and full process logs collected and verified')
