from pathlib import Path
import os,tarfile,hashlib,json,subprocess,shlex
P=Path(__file__).resolve().parent;W=P.parents[2]
archive=P/'original-build-sources.tar.gz';pins={}
with tarfile.open(archive,'w:gz') as t:
 for lib in ['flash-linear-attention-npu','ops-nn']:
  src=W/'ascend_baselines'/lib
  for root,ds,fs in os.walk(src):
   ds[:]=[d for d in ds if d not in {'.git','build','build_out','output','dist','__pycache__','.cache','tests','docs','test','node_modules','third_party_cache'}]
   for name in fs:
    p=Path(root)/name
    if p.is_symlink() or p.suffix in ['.so','.o','.a','.pyc','.run','.whl','.png','.jpg','.pdf']:continue
    rel=str(p.relative_to(W/'ascend_baselines'));pins[rel]=hashlib.sha256(p.read_bytes()).hexdigest();t.add(p,arcname=rel)
(P/'BUILD_SOURCE_PINS.json').write_text(json.dumps(pins,indent=2))
ssh=str(W/'.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh');host='openlibing-DevEnv-661495';dest='/opt/tileops-all-pairs/workspace/reports/baseline-single-agent-server1-20261027/fp64_fix/original-build'
subprocess.run([ssh,host,'mkdir -p '+shlex.quote(dest)],check=True)
with archive.open('rb') as f:subprocess.run([ssh,host,'tar -xzf - -C '+shlex.quote(dest)],stdin=f,check=True)
print('source files',len(pins),'compressed bytes',archive.stat().st_size)
