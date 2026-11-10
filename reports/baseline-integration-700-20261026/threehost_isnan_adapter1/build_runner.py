from pathlib import Path
P=Path(__file__).resolve().parent
s=(P.parent/'threehost_tla_other54_r2/run_tla.py').read_text()
s=s.replace("R=W/'.cache/all-pairs/full_rollout_20261024/remote_ready/server1/group1/recovery_resume25/tla_supplemental_1858/runtime_env2/runtime/package'","R=P/'runtime'")
s=s.replace("op=='CausalConv1dFwdOp'","op=='IsnanFwdOp'").replace('baseline700-20261026/threehost_tla_other54_r2','baseline700-20261026/threehost_isnan_adapter1').replace('from state_adapter_pto import factory','from element_adapter import factory')
s=s.replace('native.VENDOR=V;native.runtime=lambda:tilelang','native.VENDOR=V;native.runtime=lambda:tilelang;native.CACHE=P/"tla_cache"')
s=s.replace("import tilelang,torch","""import tilelang,torch
def digest(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
source=V/'testing/python/language/test_tilelang_ascend_language_elementwise.py'
assert digest(source)=='303b90f72b7b73b800f252e46cebf2dca682bea6b9362dd051f53d80044afb84'
evidence=dict(platform=os.environ['TL_PLATFORM'],runtime=tilelang.__file__,torch=torch.__version__,source_sha256=digest(source),sdk=str(sdk),files={})
for f in [sdk/'tools/bisheng_compiler/bin/bisheng',C/'handoff/bin/ptoas',*[C/'handoff/lib'/n for n in ['libtilelang.so','libtilelang_module.so','libtvm.so','libtvm_runtime.so']],R/'tilelang/jit/kernel.py',R/'tilelang/jit/adapter/cython/adapter.py']:
 if f.is_file():evidence['files'][str(f)]=dict(sha256=digest(f),inode=f.stat().st_ino)
(out/'deployment.json').write_text(json.dumps(evidence,indent=2))
""")
start=s.index('def trace(k,*args,**kwargs):')
end=s.index('JITKernel.__call__=trace',start)
s=s[:start]+'''def trace(k,*args,**kwargs):
 global seq
 seq+=1;f=Path(k.adapter.lib._name);st=f.stat();data=f.read_bytes();sha=hashlib.sha256(data).hexdigest();dest=out/'native_artifacts'/sha;dest.parent.mkdir(exist_ok=True);dest.write_bytes(data)
 r=dict(path=str(f),binary_sha256=sha,device=st.st_dev,inode=st.st_ino,symbols={},maps=Path('/proc/self/maps').read_text(),target=str(k.target),phase='before_native_call',inputs=[dict(shape=list(x.shape),dtype=str(x.dtype),numel=x.numel(),ptr=x.data_ptr()) for x in args])
 for name in ['call','init']:
  try:r['symbols'][name]=ctypes.cast(getattr(k.adapter.lib,name),ctypes.c_void_p).value
  except Exception:pass
 kernel_source=k.get_kernel_source();(out/(str(seq)+'.kernel.txt')).write_text(kernel_source);r['generated_sha256']=hashlib.sha256(kernel_source.encode()).hexdigest()
 receipt=out/(str(seq)+'.dispatch.json')
 receipt.write_text(json.dumps(r,indent=2))
 try:
  result=old(k,*args,**kwargs);r['phase']='native_returned_before_sync';receipt.write_text(json.dumps(r,indent=2))
  torch.npu.synchronize();r['phase']='native_sync_completed';r['completed']=time.time();receipt.write_text(json.dumps(r,indent=2));return result
 except Exception as e:
  r['error']=str(e);receipt.write_text(json.dumps(r,indent=2));raise
''' +s[end:]
s=s.replace(" if torch.npu.is_initialized():torch.npu.synchronize()",''' try:
  if torch.npu.is_initialized():torch.npu.synchronize()
  (out/'final_sync.json').write_text(json.dumps(dict(success=True)))
 except Exception as e:
  (out/'final_sync.json').write_text(json.dumps(dict(success=False,error=str(e))))
  raise''')
(P/'run_isnan.py').write_text(s)
compile(s,str(P/'run_isnan.py'),'exec')
