#!/usr/bin/env bash
set -euo pipefail
exec /usr/local/bin/npu-guard run --owner baseline700-20261026/sgl_conv_sampling02 --wait 3600 --timeout 300 -- /root/dyq/workspace/.tools/baselines/baseline-run sgl-kernel-npu python "$@"
