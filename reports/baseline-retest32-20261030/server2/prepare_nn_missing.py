import json,pathlib,shutil,subprocess,hashlib
R=pathlib.Path.cwd();H=R/'reports/baseline-retest32-20261030/server2';v=H/'original_nn950_extra/vendors/allpairs_nn';assert not v.exists();shutil.copytree(H/'original_nn950/vendors/allpairs_nn',v)
base=v/'op_impl/ai_core/tbe/allpairs_nn_impl';cfg=json.load(open(v/'op_impl/ai_core/tbe/config/ascend950/aic-ascend950-ops-info.json'));src=R/'reports/baseline-old1076-20261029/arm_nn_build/source';a=H/'original_nn950_extra/autogen';a.mkdir(parents=True);lines=[];receipt=[]
for op in ['LayerNormV4','BatchNormV3','BatchNormGradV3','MaskedScatter','ScatterUpdate']:
 conf=cfg[op];name=conf['opFile']['value'];found=list(src.glob('**/op_kernel/'+name+'.cpp'));assert len(found)==1,(op,found)
 parent=found[0].parent;dst=base/'ascendc'/name
 if dst.exists():raise RuntimeError('Existing source must remain immutable')
 shutil.copytree(parent,dst);lines.append('['+op+']')
 for k,vals in conf.items():
  for sub,val in vals.items():lines.append(k+'.'+sub+'='+str(val))
 lines+=['kernelSrc.value='+name,'']
 receipt.append({'op':op,'original_source':str(found[0]),'sha256':hashlib.sha256(found[0].read_bytes()).hexdigest(),'deployed_source':str(dst/found[0].name)})
ini=a/'missing.ini';ini.write_text('\n'.join(lines));g=src/'scripts/util/ascendc_impl_build.py'
with (a/'generation.log').open('w') as log:subprocess.run(['python3',str(g),str(ini),'','',str(base/'ascendc'),str(base/'dynamic'),str(a)],stdout=log,stderr=subprocess.STDOUT,check=True)
(H/'NN_MISSING_DEPLOYMENT.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
