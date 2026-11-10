#!/bin/bash
set -e
exec npu-guard run --owner baseline700-20261026/threehost_nn_grouped3_build --wait 120 --timeout 600 -- /usr/local/Ascend/ascend-toolkit/latest/tools/opbuild/op_build "$@"
