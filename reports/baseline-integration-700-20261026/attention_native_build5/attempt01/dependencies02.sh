set -e
B=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_native_build5/attempt01
export TMPDIR="$B/stage/tmp" PIP_CACHE_DIR="$B/stage/pip_cache" PYTHONDONTWRITEBYTECODE=1
P=/opt/tileops-ci/runtime/bin/python3
cp /opt/tileops-ci/python/cpython-3.12-linux-aarch64-gnu/lib/python3.12/ensurepip/_bundled/pip-25.0.1-py3-none-any.whl "$B/stage/wheels/"
export PYTHONPATH="$B/stage/wheels/pip-25.0.1-py3-none-any.whl"
"$P" -m pip download --only-binary=:all: --dest "$B/stage/wheels" build pyproject_hooks 'numpy<2.0' absl-py decorator protobuf 'scipy==1.16.3' > "$B/download02.log" 2>&1
"$P" -m pip install --no-index --find-links "$B/stage/wheels" --no-compile --target "$B/stage/python" build wheel pyproject_hooks 'numpy<2.0' absl-py decorator protobuf 'scipy==1.16.3' > "$B/dependencies_install02.log" 2>&1
sha256sum "$B"/stage/wheels/* > "$B/dependency_sha256.txt"
cat "$B/download02.log" "$B/dependencies_install02.log" "$B/dependency_sha256.txt"
