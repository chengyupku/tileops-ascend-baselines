set -e
R=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_generated_recovery5
bash "$R/cann-ops-transformer-attention_native_build5_linux-aarch64.run" --noexec --extract="$R/install_payload" > "$R/extract_package_corrected.log" 2>&1
sha256sum "$R/cann-ops-transformer-attention_native_build5_linux-aarch64.run"
python3 - <<'PY'
from pathlib import Path
import json
r=Path('/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_generated_recovery5')
(r/'PACKAGE_EXTRACT_REPAIR.json').write_text(json.dumps({'reason':'SDK makeself header accepts --extract=<path>, not upstream --target; first invocation only printed help and extracted nothing','changed_condition':'use documented --noexec --extract flag on already built original package','package_rebuilt':False,'source_changed':False},indent=2))
PY
