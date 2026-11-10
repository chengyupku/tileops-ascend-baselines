#!/usr/bin/env bash
set -eo pipefail
TASK=/root/dyq/workspace/reports/baseline-single-agent-local-20261027
source /usr/local/Ascend/cann-9.2.0-beta.2/set_env.sh
exec msprof \
  --application="timeout --foreground --signal=TERM --kill-after=5s 75s bash $TASK/local-python $TASK/run_rmsnorm_profile.py" \
  --output="$TASK/rmsnorm-profile01/msprof" \
  --runtime-api=on --task-time=on --aic-mode=sample-based \
  --aic-metrics=PipeUtilization --aic-freq=10 \
  --sys-hardware-mem=on --sys-hardware-mem-freq=10 \
  --storage-limit=200MB
