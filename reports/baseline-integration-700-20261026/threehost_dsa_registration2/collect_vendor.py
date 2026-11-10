from pathlib import Path
import subprocess,json,tarfile,io,hashlib,datetime
O=Path(__file__).resolve().parent;ssh='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
code='''import pathlib,json,tarfile,sys
P=pathlib.Path('/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_attention_moe53/frozen/TileOPs/benchmarks/ascend_all_pairs/ops_transformer/validated_manifest.json')
d=json.loads(P.read_text())['groups']['attention']['pins']
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as t:
 for n in d:
  p=pathlib.Path(n)
  if p.suffix in ('.h','.hpp','.cpp','.py','.json','.ini','.txt','.cmake','.sh','.yaml','.info') or p.name=='CMakeLists.txt':t.add(p,arcname=n.lstrip('/'),recursive=False)
'''
r=subprocess.run([ssh,'openlibing-DevEnv-055409','python3 -'],input=code.encode(),capture_output=True,timeout=55);assert r.returncode==0,r.stderr
archive=O/'ORIGINAL_SOURCE_CLOSURE.tar.gz';archive.write_bytes(r.stdout)
pins=json.loads((O/'PINS.json').read_text())['runtime'];verified={}
with tarfile.open(fileobj=io.BytesIO(r.stdout),mode='r:gz') as t:
 for member in t:
  if not member.isfile():continue
  h=hashlib.sha256(t.extractfile(member).read()).hexdigest();assert h==pins['/'+member.name],member.name;verified['/'+member.name]=h
(O/'ORIGINAL_SOURCE_CLOSURE.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),archive_sha256=hashlib.sha256(r.stdout).hexdigest(),file_count=len(verified),bytes=len(r.stdout),scope='Finite original 4671 attention pin inventory textual source/config files; no global closure search',pins=verified),indent=2)+'\n');print('source closure',len(verified),'files',len(r.stdout),'compressed bytes; all original pins matched')
