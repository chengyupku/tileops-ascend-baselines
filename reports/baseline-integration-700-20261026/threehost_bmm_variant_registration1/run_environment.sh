#!/bin/bash
set -euo pipefail
ROOT=/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_bmm_variant_registration1
unset LD_PRELOAD PYTHONHOME PYTHONPATH
# Explicit pins are serialized by the reviewed preparation script; no shell eval.
exec python3 -S "$ROOT/launch_environment.py"
