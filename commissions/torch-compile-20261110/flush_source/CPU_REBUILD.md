# Byte-identical CPU rebuild (2026-11-10)

A new CPU-only compile/link reproduced historical `flush.aibin` SHA256
`9aebeb815cc1715c3c9fe4240e98fe0fe5c360902ed88fd3f77796f82fc5ed9b`.
No device load/run was performed. This resolves a reproducible route to the historical bytes; it does not reconstruct the historical command log or prove a CANN9.1 compiler can build it.

Inputs:

- Unchanged `device_kernel.asc` in this directory.
- Public TileLang commit [0a2fb6884004b73362f584e10c35e09d9bed3a6e](https://github.com/tile-ai/tilelang/tree/0a2fb6884004b73362f584e10c35e09d9bed3a6e), `src/tl_templates/ascend`. All installed template files used in this reconstruction were byte-compared to that clean tracked tree. Individual hashes: `CPU_REBUILD_RECEIPT.json`.
- CANN **9.2.0** compiler/ASC headers, version timestamp `20261009_000323778`; compiler hash in receipt. Local compiler executable is x86_64. Recipient ARM64 compiler binary hash will differ; record it and compare final device binary before claiming equivalence.
- CANN **9.1.0** `aarch64-linux/include` for ACL host declarations. This is an explicitly recorded mixed build-input profile for the *flush helper*, not the torch.compile baseline compiler profile.

The following parameterized commands mirror the successful invocation. Use absolute paths, a fresh output directory, and quote variables. No Python device package/import is required. No compiler/source/template patches were made.

```bash
# Set these to your own existing installations; do not replace shared tools.
FLUSH_CANN92=/path/to/cann92
FLUSH_COMPILER_BIN=/path/to/cann92/tools/bisheng_compiler/bin
FLUSH_ASC=/path/to/cann92/x86_64-linux/asc   # use actual arch-specific ASC root
FLUSH_ACL91=/path/to/cann91/aarch64-linux/include
FLUSH_TILELANG=/path/to/tilelang-at-pinned-commit
FLUSH_SOURCE=/path/to/commission/flush_source
FLUSH_OUT=/path/to/fresh-build-directory
mkdir -p "$FLUSH_OUT"
ASCEND_HOME_PATH="$FLUSH_CANN92" "$FLUSH_COMPILER_BIN/bisheng" \
  -O2 -fPIC -std=c++20 -mllvm -cce-aicore-dcpreload-args=false \
  --npu-arch=dav-3510 --cce-aicore-only \
  -I "$FLUSH_TILELANG/src" -I "$FLUSH_ASC/include" -I "$FLUSH_ASC" \
  -I "$FLUSH_ACL91" "$FLUSH_SOURCE/device_kernel.asc" \
  -o "$FLUSH_OUT/flush.rel.o"
"$FLUSH_COMPILER_BIN/ld.lld" -m aicorelinux -Ttext 0 --no-mmap-output-file \
  "$FLUSH_OUT/flush.rel.o" -o "$FLUSH_OUT/flush.aibin"
sha256sum "$FLUSH_OUT/flush.aibin"
```

Build `loader.cpp` separately for the recipient host ABI with its CANN9.1 ACL headers/runtime. Typical command shape (not executed on ARM64 in this investigation):

```bash
c++ -O2 -std=c++17 -shared -fPIC loader.cpp \
  -I /path/to/cann91/aarch64-linux/include \
  -L /path/to/cann91/aarch64-linux/lib64 -lascendcl -o flush.so
```

Record actual command/compiler/linked DSO and source hashes. The loader must retain 64 blocks and 32768-byte dynamic UBUF. `init`/`call` may only execute inside the host guard. Use the already documented complete 1GiB all-ones→all-zeros and profile marker checks, then a pilot; CPU byte equality alone does not validate the recipient runtime, driver, guard, or profiler. If final device bytes differ, preserve them and investigate before claiming historical equivalence. No binaries are uploaded here.
