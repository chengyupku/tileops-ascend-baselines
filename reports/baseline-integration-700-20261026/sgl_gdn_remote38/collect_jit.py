import pathlib,json,hashlib,subprocess,shutil
P=pathlib.Path(__file__).resolve().parent;E=P/'evidence';E.mkdir(exist_ok=True);root=P/'runs/probe10--DeltaNetFwdOp--prefill-l2norm-ragged-bfloat16/output/TRITON_CACHE_DIR';rows=[]
for f in root.rglob('*'):
 if not f.is_file():continue
 if f.suffix in ['.json','.so','.o','.ttir']:
  row={'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size}
  if f.suffix=='.json':row['metadata']=f.read_text(errors='replace')
  if f.suffix in ['.so','.o']:row['elf']=subprocess.run(['readelf','-h',str(f)],capture_output=True,text=True).stdout
  rows.append(row)
  if 'chunk_local_cumsum' in f.name and f.suffix in ['.json','.ttir','.o','.so']:shutil.copyfile(f,E/('probe10_'+f.name))
S=pathlib.Path('/opt/tileops-all-pairs/build/sgl-pristine-20261025T2310/source/python/sgl_kernel_npu/sgl_kernel_npu/fla/cumsum.py');shutil.copyfile(S,E/'original_cumsum.py')
(E/'PROBE10_JIT.json').write_text(json.dumps({'artifacts':rows,'cumsum_source_sha256':hashlib.sha256(S.read_bytes()).hexdigest(),'note':'Original compiler options; no kernel or schedule edits; private attempt cache evidence.'},indent=2));print('jit artifacts',len(rows))
