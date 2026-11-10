"""CPU-only deployment layout and original adapter generation; no runtime imports."""
import pathlib,shutil,re,subprocess,hashlib,json
R=pathlib.Path.cwd();H=R/'reports/baseline-retest32-20261030/server2';src=R/'reports/baseline-old1076-20261029/arm_nn_clean_extracted/packages/vendors/allpairs_nn';dst=H/'original_nn950/vendors/allpairs_nn'
assert not dst.exists();shutil.copytree(src,dst,symlinks=False)
b=R/'reports/baseline-old1076-20261029/arm_nn_build/build';a=H/'original_nn950/autogen';a.mkdir(parents=True)
s=(b/'autogen/aic-ascend950-ops-info.ini').read_text();base=dst/'op_impl/ai_core/tbe/allpairs_nn_impl';changes=[]
chunks=re.split(r'(?=^\[)',s,flags=re.M)
for i,chunk in enumerate(chunks):
 m=re.search(r'^opFile.value=(.+)$',chunk,re.M)
 if not m:continue
 name=m[1].strip();f=base/'ascendc'/name/'arch35'/(name+'.cpp')
 if f.exists():
  line='kernelSrc.value=arch35/'+name
  chunk=re.sub(r'^kernelSrc.value=.*$',line,chunk,flags=re.M) if re.search(r'^kernelSrc.value=',chunk,re.M) else chunk.rstrip()+'\n'+line+'\n\n'
  chunks[i]=chunk;changes.append({'op_file':name,'original_source':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
cfg=a/'aic-ascend950-ops-info.ini';cfg.write_text(''.join(chunks))
gen=R/'ascend_baselines/ops-nn/scripts/util/ascendc_impl_build.py'
with (H/'original_nn950/generation.log').open('w') as log:subprocess.run(['python3',str(gen),str(cfg),'','',str(base/'ascendc'),str(base/'dynamic'),str(a)],stdout=log,stderr=subprocess.STDOUT,check=True)
manifest={'source_package':str(src),'vendor':str(dst),'generator_sha256':hashlib.sha256(gen.read_bytes()).hexdigest(),'selection_overrides':changes,'kernel_changes':False,'compiler_changes':False,'meaning':'Original generator kernelSrc metadata selects packaged arch35 implementation for Ascend950; no C++ changes'}
(H/'NN950_DEPLOYMENT.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({'selected_arch35':len(changes),'generated_dynamic':len(list((base/'dynamic').glob('*.py')))}))
