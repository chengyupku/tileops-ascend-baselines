import pathlib,json,sys,types,importlib,importlib.abc
sys.dont_write_bytecode=True
W=pathlib.Path('/opt/tileops-all-pairs/workspace');P=W/'reports/baseline-integration-700-20261026/sgl_dsa_source_reconcile1';sys.path.insert(0,str(W/'TileOPs'))
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','torch_npu','acl','triton','sgl_kernel_npu'):raise AssertionError('Forbidden real device import '+fullname)
sys.meta_path.insert(0,Block())
m=importlib.import_module('benchmarks.ascend_all_pairs.sgl.integration700_sgl_dsa_source_reconcile1')
case=types.SimpleNamespace(id=next(iter(m.KEYS)),params=m.PARAMETERS)
read=pathlib.Path.read_bytes;results=[]
for target in [P/'dsa_adapter.py']+[P/'libraries/sgl-kernel-npu'/x for x in m.PINS]:
 def changed(self):return b'changed evidence bytes' if self==target else read(self)
 pathlib.Path.read_bytes=changed
 try:
  try:m.build(case)
  except RuntimeError as e:
   assert 'changed' in str(e).lower();results.append({'path':str(target),'rejected':str(e)})
  else:raise AssertionError('Unpinned content accepted')
 finally:pathlib.Path.read_bytes=read
result={'cpu_only':True,'disk_files_mutated':False,'pin_mismatch_checks':results,'real_device_imports':0}
(P/'PIN_REJECTION_CHECK.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
