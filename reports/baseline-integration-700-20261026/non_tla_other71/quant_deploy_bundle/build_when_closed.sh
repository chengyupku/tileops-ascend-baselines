#!/usr/bin/env bash
# Intentionally non-runnable build handoff: unresolved source/SDK closure must be
# resolved and reviewed first. Do not bypass by merely changing a manifest flag.
set -euo pipefail
cat >&2 <<'EOF'
BUILD BLOCKED: source dependency closure and ops-nn 9.2.0 / CANN 9.1.0
compatibility are not proven. See DEPENDENCY_MANIFEST.json.
After resolving those blockers in an authorized private tree, the ORIGINAL
build.sh supports this invocation (not executed here):
  bash build.sh --pkg --jit --soc=ascend950 --ops=quant_batch_matmul_v3
Preserve original CMake AUTO_SYNC, OPTIONS, compiler/pass and defaults.
Do not use install3, patch GenerateEsPackage out, or copy x86 binaries.
EOF
exit 78
