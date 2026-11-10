import json,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent
for d in P.glob('case*'):
 if not d.is_dir() or not (d/'GUARD_RESULT.json').exists():continue
 c=json.loads((d/'checker.json').read_text()) if (d/'checker.json').exists() else {}
 api=c.get('api','');terms=['batchnorm','batch_norm'] if 'BatchNorm' in api else ['layernorm','layer_norm'] if ('LayerNorm' in api or 'layer_norm' in api) else ['addrmsnorm','add_rms_norm'] if 'RmsNorm' in api else ['mul']
 events=[]
 for f in (d/'ASCEND_PROCESS_LOG_PATH').rglob('*.log'):
  with f.open(errors='replace') as stream:
   for n,l in enumerate(stream,1):
    if any(t in l.lower() for t in terms) and any(t in l for t in ['allpairs_nn','allpairs_math','libops_math_layernorm','kernelName','binFileName','compile success','Register Kernel','Launch','tiling_key']):events.append(dict(file=str(f),line=n,text=l.strip()))
 (d/'NATIVE_TARGET_EVENTS.json').write_text(json.dumps(events,indent=2))
 files=[f for f in d.rglob('*') if f.is_file() and ('ASCEND_PROCESS_LOG_PATH' not in f.parts) and f.name not in ('FILES_SHA256.json',) and not any(n in f.parts for n in ('tmp','ASCEND_CACHE_PATH','ASCEND_WORK_PATH','XDG_CACHE_HOME','TORCH_EXTENSIONS_DIR'))]
 (d/'FILES_SHA256.json').write_text(json.dumps({str(f.relative_to(P)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2))
 print(d.name,'target native events',len(events))
