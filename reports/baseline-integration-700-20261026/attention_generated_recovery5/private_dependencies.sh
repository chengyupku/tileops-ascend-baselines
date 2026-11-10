set -e
R=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_generated_recovery5
O=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_native_build5
export PYTHONDONTWRITEBYTECODE=1 PIP_CACHE_DIR="$R/pip_cache"
mkdir -p "$R/wheels" "$R/python"
export PYTHONPATH="$O/attempt01/stage/wheels/pip-25.0.1-py3-none-any.whl"
/opt/tileops-ci/runtime/bin/python3 -m pip download --no-deps --only-binary=:all: --dest "$R/wheels" 'ml-dtypes==0.5.4'
/opt/tileops-ci/runtime/bin/python3 -m pip install --no-deps --no-index --find-links "$R/wheels" --target "$R/python" ml-dtypes==0.5.4
/opt/tileops-ci/runtime/bin/python3 -m pip install --no-deps --no-index --target "$R/python" "$O/attempt02/stage/source/build_out/cann_ops_transformer-1.0.0-py3-none-any.whl"
sha256sum "$R/wheels"/*
/opt/tileops-ci/runtime/bin/python3 - <<'PY'
import importlib.metadata as m
p='/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_generated_recovery5/python'
for d in m.distributions(path=[p]):print(d.metadata['Name'],d.version,d.requires)
PY
