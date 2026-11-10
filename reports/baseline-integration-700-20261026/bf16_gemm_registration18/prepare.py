import json,hashlib,pathlib,pprint,ast
O=pathlib.Path(__file__).resolve().parent; R=O.parent/'bf16_local_gemm18'; W=pathlib.Path('/root/dyq/workspace'); B=W/'TileOPs/benchmarks'
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(n,d): (O/n).write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
scope=json.loads((R/'SCOPE.json').read_text()); cases={}; envelopes={}; evidence={}
for row in scope:
 k=(row['baseline'],row['operator'],row['case_id']); p=R/k[2]/'representative.json'; r=json.loads(p.read_text()); assert (r['library'],r['op'],r['case_id'])==k
 cases[k]=json.dumps(r['case_snapshot']['parameters'],sort_keys=True,separators=(',',':'))
 a=json.loads((p.parent/'RESOURCE_ADMISSION.json').read_text());m,n,kk=a['dimensions'];e=m*kk+kk*n+m*n
 assert [a[x] for x in ('host_required','device_required','disk_required')]==[2*1024**3+e*32,2*1024**3+e*24,1024**3+e*16]
 envelopes[k[2]]={x:a[x] for x in ('dimensions','host_required','device_required','disk_required')};evidence[str(p)]=sha(p)
assert len(cases)==18
pins={}; excluded={}; provenance={}
for name in ['SOURCE_PINS.json','DEPENDENCY_FILEHASHES.json','DEPLOYMENT_MANIFEST.json','RUNTIME_FILEHASHES.json','MAPPED_RUNTIME_FILEHASHES.json']:
 d=json.loads((R/name).read_text());provenance[name]=len(d)
 for p,h in d.items():
  p=str((R/p).resolve()) if not p.startswith('/') else str(pathlib.Path(p).resolve())
  if p in pins:assert pins[p]==h,p
  pins[p]=h
# Only the original case FROZEN has benchmark/python checker dependency evidence.
f=json.loads((R/'square-1k-bfloat16/FROZEN.json').read_text())['sha256']
for p,h in f.items():
 if p.endswith('/benchmarks/ascend_all_pairs/ops_transformer/__init__.py'):
  excluded[p]={'sha256':h,'reason':'Publication/provider registry; historical checker evidence retained, not used by private native call closure.'};continue
 pins[str(pathlib.Path(p).resolve())]=h
for p in [R/'binding.py',R/'resource_gate.py',R/'capture.py',R/'SCOPE.json',R/'SOURCE_PINS.json',R/'MAPPED_RUNTIME_FILEHASHES.json',R/'DEPLOYMENT_MANIFEST.json',W/'tools/npu_guard.py',pathlib.Path('/usr/local/Ascend/cann-9.2.0-beta.2/set_env.sh')]:pins[str(p.resolve())]=sha(p)
pins.update(evidence)
for name in ('COMPILER_VERSION.txt','DEVICE_COMMAND.json','GENERATED_RECIPE_CHAIN.json','HOST_SOURCE_ABI.json','ORIGINAL_BOUNDARY.json','SOURCE_PINS.json','DEPENDENCY_FILEHASHES.json','RUNTIME_FILEHASHES.json'):
 p=R/name;pins[str(p)]=sha(p)
for p in (R/'remote_original/generated').rglob('*'):
 if p.is_file():pins[str(p)]=sha(p)
for path in ('/usr/local/Ascend/cann-9.2.0-beta.2/bin/asc_opc','/usr/local/Ascend/cann-9.2.0-beta.2/tools/bisheng_compiler/bin/bisheng'):
 p=pathlib.Path(path).resolve();pins[str(p)]=sha(p)

site=W/'.venv/lib/python3.12/site-packages'
startup=list(site.glob('*.pth'))+[site/'fla_npu_opp_env.py',pathlib.Path('/usr/bin/nm')]
for p in startup:pins[str(p.resolve())]=sha(p)

for k in cases:
 for n in ('RESOURCE_ADMISSION.json','run.sh'):p=R/k[2]/n;pins[str(p)]=sha(p)
# No broad exclusion: every excluded provider file listed with original digest.
bad=[]
for p,h in pins.items():
 if sha(p)!=h:bad.append(p)
assert not bad,bad
dump('FROZEN_PINS.json',pins);dump('PIN_EXCLUSIONS.json',excluded)
mapped={str(pathlib.Path(p).resolve()):h for p,h in json.loads((R/'MAPPED_RUNTIME_FILEHASHES.json').read_text()).items()}
local=[str(R/'resource_gate.py'),str(B/'ascend_baselines.py'),str(B/'ascend_math_more.py'),str(R/'binding.py')]
vendor=str(R/'install/vendors/bf16gmm_transformer');sdk='/usr/local/Ascend/cann-9.2.0-beta.2'
env={'ASCEND_HOME_PATH':sdk,'ASCEND_TOOLKIT_HOME':sdk,'ASCEND_AICPU_PATH':sdk,'ASCEND_OPP_PATH':sdk+'/opp','ASCEND_CUSTOM_OPP_PATH':vendor,'ASCEND_CLASSIC_TRANSFORMER_VENDOR':vendor,'ASCEND_GLOBAL_LOG_LEVEL':'0','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','PYTHONDONTWRITEBYTECODE':'1'}
head='"""CPU-lazy exact BF16 Gemm registration candidate; parent publication required."""\nimport builtins, hashlib, json, os, types, tempfile, shutil, sys\nfrom pathlib import Path\nfrom types import MappingProxyType\n'
for name,value in [('ROOT',str(R)),('MANIFEST',str(O/'FROZEN_PINS.json')),('MANIFEST_HASH',sha(O/'FROZEN_PINS.json')),('LOCAL_FILES',local),('CASE_PARAMS',cases),('ENVELOPES',envelopes),('ENV',env),('LOADED_VENDOR',mapped),('REQUIRED_LOADED',(vendor+'/op_api/lib/libcust_opapi.so',sdk+'/x86_64-linux/lib64/libnnopbase.so'))]:head+=name+' = '+pprint.pformat(value,width=120)+'\n'
base=(B/'ascend_all_pairs/ops_transformer/integration700_gdn_local_registration10.py').read_text()
helpers=base[base.index('def _overlay_mapping_matches'):base.index('def _load(data):')].replace('Frozen GDN','Frozen BF16 Gemm')
# Recheck manifest each snapshot, not merely import time.
helpers=helpers.replace('def _snapshot(runtime=False):','def _snapshot(runtime=False):\n    _checked(MANIFEST, MANIFEST_HASH)')
(O/'integration700_bf16_gemm_registration18.py').write_text(head+'\nROOT = Path(ROOT)\n'+helpers+(O/'wrapper_body.txt').read_text())
dump('CONTRACTS.json',{'status':'candidate_pending_parent_publication','case_params':[{'key':list(k),'params':json.loads(v),'envelope':envelopes[k[2]]} for k,v in cases.items()],'pin_count':len(pins),'pin_sources':provenance,'exclusions_count':len(excluded),'numeric_evidence':evidence,'algorithm':'Original binding.dense_fp32, complete aclnnGroupedMatmulV5; lossless BF16 widening/native FP32/output cast; no algorithm edits','admission':'constructor before inputs; one guard-bound token per invocation; fresh admission only if no prepared token'})
print('pins',len(pins),'excluded',len(excluded),'mapped',len(mapped))
