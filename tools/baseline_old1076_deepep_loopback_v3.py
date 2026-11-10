"""Bind the current thread's device context before original HCCL admission."""
import os,sys,runpy,ctypes
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ['DEEPEP_ALL_PAIRS_BUILD']=str(ROOT/'reports/baseline-old1076-20261029/deepep_local_build/artifacts/host/lib')
sys.path[:0]=[str(ROOT/'TileOPs'),str(ROOT/'TileOPs/src'),str(ROOT)]
os.environ['HCCL_SOCKET_IFNAME']='lo'
os.environ['HCCL_WHITELIST_DISABLE']='1'
import torch,torch_npu
import torch.distributed as dist
original_init=dist.init_process_group
def init_with_context(*args,**kwargs):
 torch.npu.set_device(0)
 acl=ctypes.CDLL('libascendcl.so')
 acl.aclrtSetDevice.argtypes=[ctypes.c_int];acl.aclrtSetDevice.restype=ctypes.c_int
 rc=acl.aclrtSetDevice(0)
 if rc:raise RuntimeError(f'Original aclrtSetDevice failed: {rc}')
 current=ctypes.c_void_p();acl.aclrtGetCurrentContext.argtypes=[ctypes.POINTER(ctypes.c_void_p)];acl.aclrtGetCurrentContext.restype=ctypes.c_int
 rc=acl.aclrtGetCurrentContext(ctypes.byref(current))
 if rc or not current.value:raise RuntimeError(f'Current device context unavailable: {rc}')
 return original_init(*args,**kwargs)
dist.init_process_group=init_with_context
runpy.run_path(str(ROOT/'tools/check_baseline_old1076_pair.py'),run_name='__main__')
