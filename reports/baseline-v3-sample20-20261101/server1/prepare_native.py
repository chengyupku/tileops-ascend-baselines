import importlib.util,pathlib,os,time,json
P=pathlib.Path(__file__).parent;R=pathlib.Path('/opt/tileops-baseline-cache/workspace');results=[]
for name,lib,kind in [('catlass','catlass','fp8_bf16'),('ops_tensor','ops-tensor','gemm')]:
 start=time.monotonic();spec=importlib.util.spec_from_file_location('cpu_build_'+name,P/'providers'/name/'build.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.WORKSPACE=R;m.VENDOR=R/'ascend_baselines'/lib;m.CACHE=R/'.cache/all-pairs'/name
 try:
  result=m.build(kind) if name=='catlass' else m.compile_native(kind)
  results.append({'library':lib,'seconds':time.monotonic()-start,'binary':str(result[0]),'status':'prepared_cpu_only'})
 except Exception as e:results.append({'library':lib,'seconds':time.monotonic()-start,'error':str(e)})
(P/'PREPARE_NATIVE.json').write_text(json.dumps(results,indent=2));print(json.dumps(results))
