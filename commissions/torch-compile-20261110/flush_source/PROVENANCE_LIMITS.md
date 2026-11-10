# Historical flush bridge evidence

These are unchanged archival text sources from the 2026-10-27 flush export.
`device_kernel.asc` SHA256: `205eee37c905de6d1fc7cd0636229a6abcb6cc81a7b802af9e97e0a505058a07`.
The recorded device binary SHA256 is `9aebeb815cc1715c3c9fe4240e98fe0fe5c360902ed88fd3f77796f82fc5ed9b` (binary not distributed).
`loader.cpp` loads that device binary through public ACL APIs: 64 blocks, 32768 bytes dynamic UBUF, same stream. It is not the generated host wrapper in host_kernel.asc.

Important correction: successful deployment on a CANN 9.1 runtime does NOT establish that this historical flush device binary was compiled with CANN 9.1. The receipt identifies a separate local compiler path and dav-3510, but does not pin its full release or template tree. Those historical build inputs/commands are still being investigated. Do not claim a verified CANN 9.1 source-build recipe from this receipt. The current installed TileLang template tree is not automatically the historical tree.

This source supplement supplies the requested host ABI bridge and device extraction for inspection; it does not resolve template/compiler compatibility or authorize modifying the device implementation. Keep flush blocked until a reproducible matching build or separately qualified identical-kernel deployment is available. No binary, new device test, or performance acceptance accompanies this supplement.
