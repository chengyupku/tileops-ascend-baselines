#!/bin/bash
set -euo pipefail
exec ssh -T -o BatchMode=yes -o ClearAllForwardings=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 openlibing-DevEnv-661495 'exec bash /opt/torch-v010-20261106/run_host_v2.sh server1'
