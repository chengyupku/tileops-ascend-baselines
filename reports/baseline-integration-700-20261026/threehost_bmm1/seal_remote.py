from pathlib import Path
import json,hashlib,re
P=Path(__file__).resolve().parent;D=P/'square-4k-bfloat16'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
f=json.loads((P/'FROZEN.json').read_text())['sha256']; changed=[p for p,h in f.items() if sha(Path(p))!=h]
exe=json.loads((D/'EXECUTION.json').read_text()); deps={p:sha(Path(p)) for p in exe['loaded_files'] if Path(p).is_file() and ('.so' in p or p.endswith('.py'))}
lines=[]
for log in (D/'ASCEND_PROCESS_LOG_PATH').rglob('*.log'):
 for line in log.read_text(errors='replace').splitlines():
  if 'GroupedMatmul' in line and ('Available bin' in line or 'Register Kernel succ' in line):lines.append(dict(log=str(log),line=line))
paths=sorted(set(re.findall(r'/\S+\.o', '\n'.join(x['line'] for x in lines))))
report=dict(frozen_changed=changed,actual_loaded_dependency_sha256=deps,device_dispatch=lines,device_binary_sha256={p:sha(Path(p)) for p in paths})
(P/'NATIVE_CLOSURE.json').write_text(json.dumps(report,indent=2))
print('frozen changed',len(changed),'loaded dependencies',len(deps),'device binaries',len(paths))
