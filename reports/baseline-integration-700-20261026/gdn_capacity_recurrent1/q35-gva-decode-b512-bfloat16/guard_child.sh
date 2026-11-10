#!/bin/bash
set -e
cp /run/lock/ascend-npu/device-0.lock /root/dyq/workspace/reports/baseline-integration-700-20261026/gdn_capacity_recurrent1/q35-gva-decode-b512-bfloat16/guard_metadata.json
exec bash /root/dyq/workspace/reports/baseline-integration-700-20261026/gdn_capacity_recurrent1/q35-gva-decode-b512-bfloat16/child.sh
