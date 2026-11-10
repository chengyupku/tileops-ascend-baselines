from pathlib import Path
import json,hashlib,subprocess,datetime
R=Path('/opt/tileops-all-pairs/workspace'); W=R/'reports/baseline-integration-700-20261026/gdn_fp16_recovery14';W.mkdir(exist_ok=True)
P=R/'.cache/all-pairs/resume22_server2_tilekernels_adaptive_norm02/sdk_parts/cann-950-ops-transformer'
files=list((P/'opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/chunk_kda_fwd').rglob('*'))+list((P/'opp/built-in/op_impl/ai_core/tbe/kernel/ascend950/ops_transformer/chunk_kda_fwd').glob('*'))+[P/'aarch64-linux/lib64/libopapi_transformer.so']+list(P.glob('**/*tiling*.so'))
rows=[]
for f in files:
 if not f.is_file():continue
 row=dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),bytes=f.stat().st_size)
 if f.suffix in ['.json','.py'] :row['text']=f.read_text()[:20000]
 if f.suffix in ['.o','.so']:
  row['elf']=subprocess.run(['readelf','-n',str(f)],capture_output=True,text=True).stdout[:2000]
  ss=subprocess.run(['strings',str(f)],capture_output=True,text=True).stdout.splitlines();row['source_strings']=[s for s in ss if ('chunk_kda' in s.lower() or 'kda_gate' in s.lower())][:120]
 rows.append(row)
rec=P/'share/info/ops_transformer/RECORD'; data=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=rows,record_matches=[s for s in rec.read_text().splitlines() if 'chunk_kda' in s.lower() or 'libopapi_transformer' in s or 'tiling.so' in s]);(W/'DEPLOYED_PROVENANCE.json').write_text(json.dumps(data,indent=2));print(json.dumps(data,indent=2))
