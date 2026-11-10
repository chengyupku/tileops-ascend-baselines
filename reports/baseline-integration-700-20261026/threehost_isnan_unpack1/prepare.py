from pathlib import Path
import ast,hashlib,json,difflib,shutil,datetime
P=Path(__file__).resolve().parent; A=P.parent/'threehost_isnan_adapter1'
s=(P/'element_adapter.parent_proposal.py').read_text(); old=(A/'element_adapter.py').read_text()
expected=old.replace('y=((y.reshape(-1,1)>>shifts)&1).reshape(shape)', 'packed=y.reshape(-1,1).expand(-1,8).contiguous()\n   expanded_shifts=shifts.reshape(1,8).expand(packed.shape[0],8).contiguous()\n   y=((packed>>expanded_shifts)&1).reshape(shape)')
assert s==expected
# Execute actual proposal expressions against CPU numpy-backed torch-like tensor interface.
import numpy as np
class Tensor:
 def __init__(self,a):self.a=np.asarray(a,dtype=np.uint8)
 @property
 def shape(self):return self.a.shape
 def reshape(self,*s):return Tensor(self.a.reshape(*s))
 def expand(self,*s):return Tensor(np.broadcast_to(self.a,tuple(a if b==-1 else b for a,b in zip(self.a.shape,s))))
 def contiguous(self):return Tensor(self.a.copy(order='C'))
 def __rshift__(self,b):return Tensor(self.a>>b.a)
 def __and__(self,b):return Tensor(self.a&b)
code='\n'.join(line.strip() for line in s.splitlines() if line.strip().startswith(('packed=','expanded_shifts=','y=((packed')))
a=np.arange(256,dtype=np.uint8).reshape(16,16);env=dict(y=Tensor(a),shifts=Tensor(np.arange(8,dtype=np.uint8)),shape=(16,128));exec(code,env)
ref=np.array([[(int(b)>>i)&1 for i in range(8)] for b in a.flat],dtype=np.uint8).reshape(16,128)
assert np.array_equal(ref,env['y'].a)
assert env['packed'].a.flags.c_contiguous and env['expanded_shifts'].a.flags.c_contiguous
assert not np.shares_memory(a,env['packed'].a)
(P/'CPU_PROOF.json').write_text(json.dumps(dict(passed=True,exhaustive_pairs=2048,rows=16,columns=128,order='packed byte-major LSB0..7; identical C-order reshape',proposal_only_expected_layout_change=True,real_device_pending=True,executed_expressions=code),indent=2))
# Observational buffer metadata only, after native wrapper has synchronized.
s=s.replace('   y=((packed>>expanded_shifts)&1).reshape(shape)', '''   (out/'unpack_buffers.json').write_text(json.dumps({n:dict(shape=list(t.shape),stride=list(t.stride()),dtype=str(t.dtype),ptr=t.data_ptr(),contiguous=t.is_contiguous()) for n,t in [('input',x),('kernel_packed_output',y),('packed',packed),('expanded_shifts',expanded_shifts)]},indent=2))
   y=((packed>>expanded_shifts)&1).reshape(shape)''')
(P/'element_adapter.py').write_text(s)
(P/'adapter.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='previous_adapter',tofile='expanded_adapter')))
r=(A/'run_isnan.py').read_text().replace('threehost_isnan_adapter1','threehost_isnan_unpack1')
r=r.replace("r['completed']=time.time();", "r['output']=dict(shape=list(result.shape),stride=list(result.stride()),dtype=str(result.dtype),ptr=result.data_ptr());r['completed']=time.time();")
r=r.replace("return original_subprocess_run(cmd,*args,**kwargs)", """ headers=json.loads((P/'headers_manifest.json').read_text())
  assert all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in headers.items()), 'header drift'
  (out/'headers_manifest.json').write_text(json.dumps(headers,indent=2))
  for f in cmd:
   if str(f).endswith('.cpp') and Path(f).is_file(): (out/'compiled.cpp').write_bytes(Path(f).read_bytes())
 return original_subprocess_run(cmd,*args,**kwargs)""")
r=r.replace("'BISHENG_HOME']}),", "'BISHENG_HOME','PATH','LD_LIBRARY_PATH','PYTHONPATH','TILELANG_LIBRARY_PATH','TVM_LIBRARY_PATH','ASCEND_TOOLKIT_HOME','TLA_CYTHON_CACHE_DIR']}),")
r=r.replace("(out/'deployment.json').write_text", "evidence['loaded_modules']={n:str(getattr(m,'__file__','')) for n,m in sys.modules.items() if n.split('.')[0] in ('torch','torch_npu','tilelang','tvm')}\n(out/'deployment.json').write_text")
compile(r,'run_isnan.py','exec');(P/'run_isnan.py').write_text(r)
for n in ['SCOPE.json','original_source.py','compare.original.txt']:shutil.copyfile(A/n,P/n)
(P/'HEARTBEAT.json').write_text(json.dumps(dict(phase='cpu_equivalence_passed',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),accepted_added=0)))
print('CPU exhaustive layout proof passed; runner syntax passed')
