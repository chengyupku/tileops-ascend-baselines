"""Host-side validation for the unchanged exported unified flush kernel."""
import ctypes
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def install_exported_flush(torch):
    """Deploy the identical exported flush kernel on a host without TileLang JIT.

No source/schedule changes: the checked-in generated 1GiB kernel is compiled
for the remote host ABI; measurement still uses the unmodified unified core,
including its profiled marker validation and per-invocation flush/sync.
"""
    directory = ROOT / 'reports/baseline-cache-20261027/flush'
    library = directory / 'flush.so'
    if not library.exists():
        return None
    source = json.loads((directory / 'SOURCE.json').read_text())
    if hashlib.sha256((directory / 'host_kernel.asc').read_bytes()).hexdigest() != source['generated_source_sha256']:
        raise ValueError('Exported flush source changed')
    from tools import measure_baselines as core
    dll = ctypes.CDLL(str(library))
    dll.init.argtypes, dll.init.restype = [ctypes.c_char_p], ctypes.c_int
    dll.call.argtypes, dll.call.restype = [ctypes.c_void_p, ctypes.c_void_p], ctypes.c_int
    dll.get_last_error.argtypes, dll.get_last_error.restype = [], ctypes.c_char_p
    if hashlib.sha256((directory / 'flush.aibin').read_bytes()).hexdigest() != source['device_binary_sha256']:
        raise ValueError('Exported flush device binary changed')
    if dll.init(str(directory / 'flush.aibin').encode()) != 0:
        raise RuntimeError('Exported original flush initialization failed')
    def flush(buffer):
        rc = dll.call(buffer.data_ptr(), torch.npu.current_stream().npu_stream)
        if rc:
            raise RuntimeError(str(dll.get_last_error()))
    buffer = torch.ones(268435456, dtype=torch.int32, device='npu')
    flush(buffer)
    torch.npu.synchronize()
    # Validate every byte on CPU: expanded native OPP overrides device ReduceSum.
    # This is outside every profiling window and does not change the flush kernel.
    host_buffer = buffer.cpu()
    if host_buffer.numel() != 268435456 or torch.count_nonzero(host_buffer).item() != 0:
        raise ValueError('Original exported flush did not cover the full 1GiB buffer')
    del host_buffer
    core._flush_kernel_cache[torch.npu.current_device()] = (flush, buffer)
    return {**source, 'library_sha256': hashlib.sha256(library.read_bytes()).hexdigest(), 'whole_buffer_check': '268435456 int32 ones all overwritten with zero', 'whole_buffer_check_location': 'CPU copy of complete device buffer before profiling'}
