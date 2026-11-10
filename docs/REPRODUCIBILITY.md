# Portability and measurement

The repository preserves source versions rather than silently rewriting original
kernels, reference implementations, or checkers for a new host. Historical
launchers contain deployment-specific `/opt/...` and workspace paths. Do not run
them as installation scripts without configuring the paths and reviewing their
scope. No GitHub Actions workflow is installed.

A new execution environment must bind the exact canonical case contract, adapter
version, original library source and matching compiled artifact. Record physical
AIC/AIV configuration, CANN/driver/compiler, torch/torch_npu, library commit and
binary SHA, timing parser, and actual checker. A matching Python source file does
not establish the provenance or API coverage of a shared library.

The recent accepted campaigns used physical 32 AIC/64 AIV, three warmups, fifteen
measurement windows, 1 GiB flush, valid foreground npu-guard reservations, unchanged
canonical correctness checks and full raw-window replay. Optional adaptations
must count metadata/layout/rotation/output conversion inside the timed operation.
The permitted native route is a complete public API or at most three existing
native operations; a source file being included does not establish compliance.

All imports that may initialize NPU and all device work must be wrapped in the
host's established npu-guard foreground command. Respect other tasks and health/
visibility checks. This export does not authorize resuming the paused campaign.

`patches/` records tracked text differences from upstream HEAD as source evidence.
Do not automatically apply them to claim an original-kernel baseline. Check each
route's provenance. Untracked generated binaries and caches are not exported.
Submodule checkout, patch review, host-specific dependency builds, and numerical
validation remain distinct stages.
