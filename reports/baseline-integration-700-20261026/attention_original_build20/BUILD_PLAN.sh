#!/usr/bin/env bash
# NOT EXECUTED. Run on server2 only after supervisor authorizes build/deployment.
# This script stages unchanged original source and only writes a new private subtree.
set -euo pipefail
SRC=/opt/tileops-all-pairs/workspace/.cache/all-pairs/root_server2_20261022/source_templates/ascend_baselines/ops-transformer
STAGE=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_original_build20/build_stage
PY=/opt/tileops-ci/runtime/bin/python3
SDK=/usr/local/Ascend/cann-9.1.0
# Refuse reuse rather than overwrite any existing build evidence.
test ! -e "$STAGE"
mkdir -p "$STAGE/source" "$STAGE/python" "$STAGE/wheels" "$STAGE/install"
tar -C "$SRC" --exclude='./build' --exclude='./build_out' --exclude='./.git' -cf - . | tar -C "$STAGE/source" -xf -
source "$SDK/set_env.sh"
export ASCEND_HOME_PATH="$SDK"
export ASCEND_CANN_PACKAGE_PATH="$SDK"
export PYTHONDONTWRITEBYTECODE=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONPATH="$STAGE/python:/opt/tileops-ci/runtime/lib/python3.12/site-packages"
export TORCH_EXTENSIONS_DIR="$STAGE/torch_extensions"
# Environment repair is private, no shared dependency install. Download/install was NOT done by worker.
"$PY" -m pip download --dest "$STAGE/wheels" build wheel setuptools packaging
"$PY" -m pip install --no-index --find-links "$STAGE/wheels" --target "$STAGE/python" build wheel setuptools packaging
mkdir -p "$STAGE/bin"
ln -s "$PY" "$STAGE/bin/python3"
export PATH="$STAGE/bin:$PATH"
cd "$STAGE/source"
# Official two-op native build; metadata native operator is mandatory.
# Optional cann-cmake fetch uses pinned master-058, see EVIDENCE.json.
bash build.sh --pkg --soc=ascend950 --ops=flash_attn,flash_attn_metadata --vendor_name=attention_build20 -j8
mapfile -t runs < <(find "$STAGE/source/build_out" -maxdepth 1 -name '*.run' -type f)
test "${#runs[@]}" -eq 1
bash "${runs[0]}" --install-path="$STAGE/install"
# Full wrapper wheel retains official cann_ops_transformer import name; native subset above remains scoped.
bash build.sh --torch_extension --soc=ascend950
mapfile -t wheels < <(find "$STAGE/source/build_out" -maxdepth 1 -name 'cann_ops_transformer-*.whl' -type f)
test "${#wheels[@]}" -eq 1
"$PY" -m pip install --no-deps --target "$STAGE/python" "${wheels[0]}"
test -f "$STAGE/install/vendors/attention_build20_transformer/bin/set_env.bash"
# Do not import wrappers or invoke firstcase here. OpBuilder loads/JIT-compiles native binding lazily.
# Later reserved checker must source the vendor environment above, use these PYTHONPATH/
# TORCH_EXTENSIONS_DIR settings, import firstcase_adapter, and preserve full original oracle.
