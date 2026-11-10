"""CPU-only full task-identity verification for long single-stream captures."""
from collections import Counter
import hashlib
import struct


def verify_capture(root, native_marker, expected_native):
    hashes = {}
    files = {}
    for path in root.rglob('*hash_dic.slice_*'):
        if path.name.endswith('.done'):
            continue
        data = path.read_bytes()
        files[str(path.relative_to(root))] = hashlib.sha256(data).hexdigest()
        for line in data.decode(errors='replace').splitlines():
            key, sep, name = line.partition(':')
            if sep:
                key = int(key)
                hashes[key if key < 2**63 else key - 2**64] = name
    tracked = {}
    for path in root.rglob('*task_track.slice_*'):
        if path.name.endswith('.done'):
            continue
        data = path.read_bytes()
        files[str(path.relative_to(root))] = hashlib.sha256(data).hexdigest()
        assert len(data) % 64 == 0, 'Partial host task record'
        for offset in range(0, len(data), 64):
            values = struct.unpack_from('<8q', data, offset)
            if values[5]:
                key = (values[3] >> 32) & 0xFFFFFFFF
                assert key not in tracked, 'Duplicate complete task identity'
                assert values[5] in hashes, 'Unresolved kernel name'
                tracked[key] = hashes[values[5]]
    device = Counter()
    for path in root.rglob('ffts_profile*'):
        if path.name.endswith('.done'):
            continue
        data = path.read_bytes()
        files[str(path.relative_to(root))] = hashlib.sha256(data).hexdigest()
        assert len(data) % 128 == 0, 'Partial device record'
        for offset in range(0, len(data), 128):
            values = struct.unpack_from('<16q', data, offset)
            if values[1] and values[15] >= values[14]:
                device[(values[0] >> 32) & 0xFFFFFFFF] += 1
    assert device == Counter(tracked.keys()), 'Host/device task identities do not match exactly'
    names = Counter(tracked.values())
    native_count = sum(count for name, count in names.items() if native_marker in name)
    assert native_count == expected_native, (native_count, expected_native, names)
    assert not any('_tilelang_profiler_cache_flush' in name for name in names), 'Flush leaked into timed window'
    return {'identity_bits': 32, 'native_records': native_count,
            'all_device_records': sum(device.values()), 'named_kernel_counts': dict(names),
            'truncated_16bit_collisions': len(tracked) - len({key & 0xFFFF for key in tracked}),
            'capture_sha256': files}
