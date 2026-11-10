import pathlib,json,hashlib,re,subprocess,shutil
P=pathlib.Path(__file__).resolve().parent; S=pathlib.Path('/opt/tileops-all-pairs/build/sgl-pristine-20261025T2310/source');E=P/'evidence';E.mkdir(exist_ok=True)
files=[S/'python/sgl_kernel_npu/sgl_kernel_npu/fla/chunk.py',S/'python/sgl_kernel_npu/sgl_kernel_npu/__init__.py']; hashes={}
for f in files:
 target=E/('original_'+f.parent.name+'_'+f.name);shutil.copyfile(f,target);hashes[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
objects={}; events=[]
for log in (P/'runs').glob('*/output/native.jsonl'):
 for line in log.open():
  d=json.loads(line);s=d['text']
  if 'StreamLaunchKernel' in s and any(k in s.lower() for k in ['matmul','bmm','isfinite','cumsum']):
   if len(events)<16:events.append({'receipt':str(log.relative_to(P)),**d})
  m=re.search(r'bin path[: ]+([^ ,]+\.o)',s)
  if m and any(k in m[1].lower() for k in ['matmul','bmm','isfinite','cumsum']):objects[m[1]]=None
for name in list(objects)[:8]:
 f=pathlib.Path(name)
 if f.exists():objects[name]={'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size,'elf_header':subprocess.run(['readelf','-h',name],capture_output=True,text=True).stdout}
lib=S/'python/sgl_kernel_npu/sgl_kernel_npu/lib/libsgl_kernel_npu.so';hashes[str(lib)]=hashlib.sha256(lib.read_bytes()).hexdigest()
(E/'NATIVE_EVIDENCE.json').write_text(json.dumps({'source_hashes':hashes,'launch_samples':events,'kernel_objects':objects,'sgl_elf':subprocess.run(['readelf','-h',str(lib)],capture_output=True,text=True).stdout,'note':'Original chunk_gated_delta_rule_native is an upstream complete torch-based API; CANN launches shown, not claimed custom SGL kernel calls.'},indent=2))
root=pathlib.Path('/opt/tileops-all-pairs/workspace/.cache/all-pairs/remote_native_vllm/python-deps/fla');(E/'REMOTE_FLA_HASHES.json').write_text(json.dumps({str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in root.rglob('*.py')},indent=2))
print('collected evidence')
