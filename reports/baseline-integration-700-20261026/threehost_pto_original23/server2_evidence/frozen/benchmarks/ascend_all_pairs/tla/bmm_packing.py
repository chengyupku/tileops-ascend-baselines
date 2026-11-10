"""Fail-closed compensation for individually verified original Bmm dispatches.

This is an external A bit permutation, not a kernel change. Manifest cases absent
from the whitelist are NOT numerically accepted. Source/template/ELF drift must
be reviewed, especially when upstream changes L0A from zZ to NZ.
"""
import ctypes
import hashlib
import json
import struct
from pathlib import Path


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _device_elf_hashes(data):
    """Read complete embedded ET_EXEC AI-core ELFs; do not load the library."""
    result = []
    pos = 1
    while True:
        pos = data.find(b'\x7fELF', pos)
        if pos < 0:
            return sorted(result)
        if data[pos + 4:pos + 6] == b'\x02\x01':
            h = struct.unpack_from('<16sHHIQQQIHHHHHH', data, pos)
            if h[1:3] == (2, 4137):
                phoff, shoff, ehsize = h[5], h[6], h[8]
                phentsize, phnum, shentsize, shnum = h[9:13]
                size = max(ehsize, phoff + phentsize * phnum, shoff + shentsize * shnum)
                for i in range(shnum):
                    section = struct.unpack_from('<IIQQQQIIQQ', data, pos + shoff + i * shentsize)
                    if section[1] != 8:  # SHT_NOBITS occupies no file bytes.
                        size = max(size, section[4] + section[5])
                if size <= 64 or pos + size > len(data):
                    raise ValueError('Invalid embedded device ELF extent')
                result.append(hashlib.sha256(data[pos:pos + size]).hexdigest())
        pos += 4


def _validate_metadata(metadata, admitted):
    for key in ('case_id', 'args', 'dtype', 'soc', 'soc_enum', 'target',
                'execution_backend', 'platform', 'target_host', 'pass_configs', 'compile_flags',
                'source_sha256', 'device_elf_sha256'):
        if metadata[key] != admitted[key]:
            raise ValueError('Bmm physical-layout gate rejected ' + key)


def pack_bmm_a_zz_for_3510(a, torch):
    if a.ndim != 3 or a.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError('Only original 3D FP16/BF16 A is admitted')
    batch, m, k = a.shape
    if min(batch, m, k) <= 0 or m % 128 or k % 64 or not a.is_contiguous():
        raise ValueError('Expected contiguous A padded to original 128x64 tiles')
    physical_nz = a.reshape(batch, m // 128, 8, 16, k // 64, 4, 16).permute(0, 1, 4, 5, 2, 3, 6).contiguous()
    return physical_nz.reshape(batch, m // 128, k // 64, 8, 4, 16, 16).permute(0, 1, 3, 5, 2, 4, 6).reshape(batch, m, k).contiguous()


def packing_for(kernel, case_id, args, dtype, vendor):
    """Called inside the normal guarded original factory after compilation."""
    from benchmarks.ascend_baselines import UnsupportedCase
    try:
        evidence = json.loads(Path(__file__).with_name('bmm_packing_identity.json').read_text())
        if case_id not in evidence['admitted']:
            raise ValueError('Original Bmm case has not passed this compensation')
        # Explicit hashes prevent double compensation after any upstream layout fix.
        for relative, expected in evidence['vendor_sha256'].items():
            if _sha(Path(vendor) / relative) != expected:
                raise ValueError('Bmm template/builder changed: ' + relative)
        for path, expected in evidence['toolchain_sha256'].items():
            if _sha(path) != expected:
                raise ValueError('Bmm toolchain/platform changed: ' + path)
        import torch_npu
        acl = ctypes.CDLL('libascendcl.so')
        acl.aclrtGetSocName.restype = ctypes.c_char_p
        soc = acl.aclrtGetSocName().decode()
        library = Path(kernel.adapter.lib._name)
        metadata = dict(case_id=case_id, args=list(args), dtype=dtype,
                        soc=soc, soc_enum=torch_npu.npu.get_soc_version(),
                        target=str(kernel.target), execution_backend=kernel.execution_backend,
                        platform=str(kernel.platform), target_host=str(kernel.target_host),
                        pass_configs={getattr(k, 'value', str(k)): v for k, v in kernel.pass_configs.items()},
                        compile_flags=kernel.compile_flags,
                        source_sha256=hashlib.sha256(kernel.get_kernel_source().encode()).hexdigest(),
                        device_elf_sha256=_device_elf_hashes(library.read_bytes()))
        _validate_metadata(metadata, evidence['admitted'][case_id])
    except (ValueError, KeyError, OSError, struct.error) as exc:
        raise UnsupportedCase(str(exc)) from exc
    import torch
    def pack(a):
        if tuple(a.shape) != (args[0], args[1], args[3]) or a.dtype != getattr(torch, dtype) or a.device.type != 'npu':
            raise UnsupportedCase('Bmm runtime A differs from admitted dispatch')
        return pack_bmm_a_zz_for_3510(a, torch)
    return pack
