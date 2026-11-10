#!/usr/bin/env bash
set -euo pipefail
prefix=/opt/tileops-ci
cd "$prefix/actions-runner"
export RUNNER_ALLOW_RUNASROOT=1
# This lock prevents duplicate CPU-only runner listeners. Device reservations
# remain the separate, administrator-provided host NPU lock.
exec 9> "$prefix/runner-listener.lock"
flock -n 9 || exit 0
exec ./run.sh
