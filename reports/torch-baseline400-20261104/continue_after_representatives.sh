#!/bin/bash
set -euo pipefail
while systemctl is-active --quiet torch-baseline400-server1.service; do sleep 5; done
exec bash /root/dyq/workspace/reports/torch-baseline400-20261104/server1_v3_service.sh
