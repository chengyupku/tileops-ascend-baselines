import ctypes as C
import os
from pathlib import Path
WORKSPACE=Path(__file__).resolve().parent.parent
def _owned_native(path, symbols):
 lib=C.CDLL(str(path))
 for name in symbols:getattr(lib,name)
 return lib
class _MathCall:
    def __init__(self, api, arguments, *, library="ops-math", host_library=None):
        import torch

        self.torch = torch
        self.pending = []
        suffix = {"ops-math": "math", "ops-nn": "nn", "ops-transformer": "transformer"}[library]
        self.vendor = Path(os.environ.get(
            f"ASCEND_CLASSIC_{suffix.upper()}_VENDOR",
            str(WORKSPACE / ".cache/ascend-classic/native-pilot-20261020" / library
                / f"install/vendors/classicpilot_{suffix}"),
        )).resolve()
        configured = {Path(p).resolve() for p in
                      os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(":") if p}
        if self.vendor not in configured and api != "aclStft":
            raise RuntimeError(f"Select the isolated {library} vendor before importing Torch.")
        self.binary = (Path(host_library).resolve() if host_library else
                       self.vendor / "op_api/lib/libcust_opapi.so")
        self.lib = _owned_native(self.binary, [api, api + "GetWorkspaceSize"])
        self.api = api
        self.size = getattr(self.lib, api + "GetWorkspaceSize")
        self.size.argtypes = arguments + [C.POINTER(C.c_uint64), C.POINTER(C.c_void_p)]
        self.size.restype = C.c_int
        self.launch = getattr(self.lib, api)
        self.launch.argtypes = [C.c_void_p, C.c_uint64, C.c_void_p, C.c_void_p]
        self.launch.restype = C.c_int
        self.base = C.CDLL("libnnopbase.so")
        self.base.aclCreateTensor.argtypes = [
            C.POINTER(C.c_int64), C.c_uint64, C.c_int, C.POINTER(C.c_int64),
            C.c_int64, C.c_int, C.POINTER(C.c_int64), C.c_uint64, C.c_void_p,
        ]
        self.base.aclCreateTensor.restype = C.c_void_p
        self.base.aclDestroyTensor.argtypes = [C.c_void_p]
        self.base.aclDestroyTensor.restype = C.c_int
        self.base.aclCreateIntArray.argtypes = [C.POINTER(C.c_int64), C.c_uint64]
        self.base.aclCreateIntArray.restype = C.c_void_p
        self.base.aclDestroyIntArray.argtypes = [C.c_void_p]
        self.base.aclDestroyIntArray.restype = C.c_int
        self.codes = {torch.float32: 0, torch.float16: 1, torch.int32: 3,
                      torch.int64: 9, torch.complex64: 16, torch.bfloat16: 27,
                      torch.float8_e4m3fn: 36}

    @staticmethod
    def check(rc, api):
        if rc:
            raise RuntimeError(f"{api} returned {rc}")

    def record(self):
        item = {"tensors": [], "arrays": [], "keep": []}
        self.pending.append(item)
        return item

    def tensor(self, tensor, record, *, format_code=2):
        if not tensor.is_contiguous():
            raise ValueError("This host bridge expects packed tensors.")
        shape = (C.c_int64 * tensor.ndim)(*tensor.shape)
        stride = (C.c_int64 * tensor.ndim)(*tensor.stride())
        record["keep"].extend((tensor, shape, stride))
        ptr = self.base.aclCreateTensor(shape, tensor.ndim, self.codes[tensor.dtype],
                                       stride, 0, format_code, shape, tensor.ndim, tensor.data_ptr())
        if not ptr:
            raise RuntimeError("aclCreateTensor returned null")
        record["tensors"].append(ptr)
        return ptr

    def array(self, values, record):
        storage = (C.c_int64 * len(values))(*values)
        record["keep"].append(storage)
        ptr = self.base.aclCreateIntArray(storage, len(values))
        if not ptr:
            raise RuntimeError("aclCreateIntArray returned null")
        record["arrays"].append(ptr)
        return ptr

    def call(self, arguments, record, device):
        size, executor = C.c_uint64(), C.c_void_p()
        self.check(self.size(*arguments, C.byref(size), C.byref(executor)),
                   self.api + "GetWorkspaceSize")
        workspace = self.torch.empty(size.value, device=device, dtype=self.torch.uint8)
        record["keep"].extend((workspace, executor))
        self.check(self.launch(workspace.data_ptr() if size.value else None,
                               size.value, executor,
                               self.torch.npu.current_stream().npu_stream), self.api)

    def cleanup(self):
        self.torch.npu.synchronize()
        while self.pending:
            record = self.pending.pop()
            for p in record["tensors"]:
                self.check(self.base.aclDestroyTensor(p), "aclDestroyTensor")
            for p in record["arrays"]:
                self.check(self.base.aclDestroyIntArray(p), "aclDestroyIntArray")

    def hashes(self, root):
        sources = [p for p in root.rglob("*")
                   if p.suffix in (".cpp", ".h", ".cc", ".py", ".json")]
        deployed = [p for p in self.vendor.rglob("*")
                    if p.is_file() and p.suffix in (".so", ".o", ".json")]
        return source_hashes(Path(__file__), self.binary, *sources, *deployed)

