import json,hashlib
from pathlib import Path
O=Path('/root/dyq/workspace/reports/baseline-integration-700-20261026/gdn_chunk_registration16')
ns={};exec(compile((O/'integration700_gdn_chunk_registration16.py').read_bytes(),'frozen','exec'),ns)
header=next(Path(p) for p in ns['RUNTIME_PINS'] if '/source_templates/' in p)
root=header.parent.parent.parent
files=[]
for folder in ('op_host','op_kernel'):
 for p in sorted((root/folder).rglob('*')):
  if not p.is_file() or p.suffix not in ('.cpp','.h'):continue
  raw=p.read_bytes();lines=raw.decode(errors='replace').splitlines();hits=[{'line':i,'text':s} for i,s in enumerate(lines,1) if any(w in s for w in ('ADD_TO_LAUNCHER_LIST_AICORE','aclnnChunkGatedDeltaRule','ChunkGatedDeltaRule','AICORE','tilingKey','TILING_KEY_IS'))]
  files.append({'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'lines':hits[:35]})
metadata=[]
for p,pin in ns['RUNTIME_PINS'].items():
 if '/chunk_binary/' in p and p.endswith('.json'):
  raw=Path(p).read_bytes();assert hashlib.sha256(raw).hexdigest()==pin;metadata.append({'path':p,'sha256':pin,'data':json.loads(raw)})
result={'source_root':str(root),'sources':files,'metadata':metadata,'boundary':'Source and candidate metadata only. Original loaded-opapi maps support API ownership, not per-case binary selection; no new execution.'}
(O/'NATIVE_SOURCE_METADATA.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
